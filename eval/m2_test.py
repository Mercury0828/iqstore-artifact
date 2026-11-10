"""M2 TEST evaluation (prereg/M2.md v4 §5): Q1 / H-a storage and H-e correctness. Runs only after the DEV freeze.

Guard: `prereg/M2_dev_freeze.json` must exist, be tracked by git and unmodified (its sha256 is printed and stored in every
output). TEST epochs are read through `eval/m2_data.py` with allow_test=True.

  q1   per TEST dataset: b-bar of every storage method (IQSTORE at the frozen settings with its ingest-time search,
       B4q, B4, A7, B2-zfp, B2-SZ3; B0, B1) for families F-boot, F-plan, F-drift and tau in {1e-2, 1e-3, 1e-4}; IQSTORE
       round trip and component bytes [F12]. Output runs/m2_test/q1/<dataset>.json (resumable).
  he   per TEST dataset (needs every later epoch; run after the campaign ends): the H-e allocation of M2_workload.json
       (class, vector source and selector regenerated from STREAM 4 and checked against he_stream_sha256), answered by
       IQSTORE (tau = 1e-3) under F-boot and under F-drift: exact interval (Fractions) vs the exact raw answer; checks
       containment F- <= F <= F+ and the T1 width bound F+ - F- <= osc(g) * #(shots with an ambiguous position of S).
       Output runs/m2_test/he/<dataset>_<family>.json.

Usage: python eval/m2_test.py q1|he [--shard k/n]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import m2_data as M  # noqa: E402
from eval.workload import rng  # noqa: E402

OUT = ROOT / "runs" / "m2_test"
TAUS = (1e-2, 1e-3, 1e-4)
FAMILIES = ("F-boot", "F-plan", "F-drift")
FAM_HE = ("F-boot", "F-drift")


def freeze() -> tuple[dict, str]:
    f = M.FREEZE
    if not f.exists():
        raise PermissionError("no DEV freeze")
    rel = f.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("the DEV freeze is not committed unmodified")
    b = f.read_bytes()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def test_datasets(families: bool = True):
    w = M.workload()
    accepted = M.ledger_jobs()
    eps = {}
    for entry in w["datasets"]:
        if entry["split"] != "TEST":
            continue
        e = int(entry["campaign_day"])
        if e not in accepted:
            continue  # epoch not collected yet
        if e not in eps:
            eps[e] = M.load_epoch(e, allow_test=True)
        yield entry, eps[e], M.load_dataset(entry, eps[e], families)


def fdrift_families(ds, fz: dict):
    from iqstore.family import family_from_widths
    c = fz["F-drift"]
    return [family_from_widths(p.cal.ref, c[p.context]["dtheta_half_width"], c[p.context]["dbeta_half_width"])
            for p in ds.positions]


# ---------------------------------------------------------------------------------------------------- Q1

def q1(shard=(0, 1)) -> None:
    from eval.m2_storage import lossless_row, storage_row
    fz, sha = freeze()
    iq = fz["IQSTORE"]
    out = OUT / "q1"
    out.mkdir(parents=True, exist_ok=True)
    store = OUT / "containers"
    store.mkdir(parents=True, exist_ok=True)
    for i, (entry, ep, ds) in enumerate(test_datasets()):
        f = out / f"{ds.name}.json"
        if f.exists() or i % shard[1] != shard[0]:
            continue
        t0 = time.time()
        row = {"freeze_sha256": sha, "epoch": ds.epoch, "m": len(ds.positions), "N": ds.positions[0].raw.size} | \
            lossless_row(ds, store)
        for fname in FAMILIES:
            fams = fdrift_families(ds, fz) if fname == "F-drift" else None
            for tau in TAUS:
                row[f"{fname}|{tau:g}"] = storage_row(ds, fname, tau, iq, store, families=fams,
                                                      methods=("IQSTORE", "B4q", "B4", "A7", "B2-zfp", "B2-SZ3"))
        row["seconds"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(row, indent=1), encoding="utf-8")
        r = row["F-boot|0.001"]
        print(ds.name, row["seconds"], "s; F-boot 1e-3 IQ %.4f B4q %.4f B2 %.3f/%.3f B0 %.1f" %
              (r["IQSTORE"], r["B4q"], r["B2-zfp"], r["B2-SZ3"], row["B0"]), flush=True)


# ---------------------------------------------------------------------------------------------------- H-e

def he_allocation(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regenerates (class, source, selector) exactly as eval/workload.py does."""
    n = int(entry["he_allocation"]["n_pairs"])
    g = rng("he_alloc", int(entry["index"]))
    cls = g.integers(1, 6, n)
    src = g.choice(np.array([2, 1, 3]), n, p=[0.5, 0.25, 0.25])
    sel = g.integers(0, 1 << 30, n)
    src[cls == 5] = 0
    return cls, src, sel


def check_he_digest() -> bool:
    w = M.workload()
    d = hashlib.sha256()
    for e in w["datasets"]:
        if e["split"] == "TEST":
            cls, src, sel = he_allocation(e)
            d.update(cls.astype(np.int8).tobytes() + src.astype(np.int8).tobytes() + sel.astype(np.int64).tobytes())
    return d.hexdigest() == w["he_stream_sha256"]


def _vec_from_uv(fam, ref, u: float, v: float):
    from iqstore.family import Discriminator
    return Discriminator(fam.th_lo + u * (fam.th_hi - fam.th_lo), fam.b_lo + v * (fam.b_hi - fam.b_lo))


def he(shard=(0, 1)) -> None:
    from eval.m2_cov import later_refs_all, queries
    from iqstore.ingest import ingest_position
    from iqstore.query import exact_labels, exact_values_multi, g_count, g_parity, intervals_multi
    from iqstore.rep import LS_BOTH
    from iqstore.views import assignment_matrix, mitigation_g

    fz, sha = freeze()
    if not check_he_digest():
        raise AssertionError("H-e allocation does not reproduce he_stream_sha256")
    iq = fz["IQSTORE"]
    out = OUT / "he"
    out.mkdir(parents=True, exist_ok=True)
    w = M.workload()
    for i, (entry, ep, ds) in enumerate(test_datasets()):
        if i % shard[1] != shard[0]:
            continue
        qs = queries(entry, ds, ep)
        refs = later_refs_all(ds)  # {later epoch: [h per position]} over every accepted later epoch
        cls, src, sel = he_allocation(entry)
        for fname in FAM_HE:
            f = out / f"{ds.name}_{fname}.json"
            if f.exists():
                continue
            t0 = time.time()
            fams = fdrift_families(ds, fz) if fname == "F-drift" else [p.fams["F-boot"] for p in ds.positions]
            reps = []
            for p, fam in zip(ds.positions, fams):
                x, y = p.cal.frame.apply(p.raw)
                reps.append(ingest_position(p.cal.frame, fam, x, y, p.raw, 1e-3, iq["d0"], iq["max_depth"],
                                            iq["split_fraction"])[0])
            frames = [p.cal.frame for p in ds.positions]
            raws = [p.raw for p in ds.positions]
            vec = entry["vectors"]
            later = sorted(refs)
            stats = {"pairs": 0, "statistics": 0, "violations": [], "width_bound_violations": [],
                     "b_fallback_to_c": 0, "by_class": {}}
            ls_cache, lab_cache = {}, {}

            def lsets(j, h):
                key = (j, h.theta, h.beta)
                if key not in ls_cache:
                    ls_cache[key] = reps[j].label_set_codes(h)
                    lab_cache[key] = exact_labels([raws[j]], [frames[j]], [h])[:, 0]
                return ls_cache[key], lab_cache[key]

            for c, s, z in zip(cls.tolist(), src.tolist(), sel.tolist()):
                if c == 5:
                    pairs = qs["2"]
                    inst = z % (len(pairs) * 50)
                    S = pairs[inst // 50]
                    pt = inst % 50
                    u = entry["queries"]["5_sweep"]["u"][pt // 10]
                    v = entry["queries"]["5_sweep"]["v"][pt % 10]
                    hs = [_vec_from_uv(fams[j], ds.positions[j].cal.ref, u, v) for j in S]
                    stat_kind = "2"
                else:
                    insts = qs[str(c)]
                    if not insts:
                        continue
                    S = insts[z % len(insts)]
                    stat_kind = str(c)
                    s_eff = s
                    if s == 1 and not later:
                        s_eff = 2
                        stats["b_fallback_to_c"] += 1
                    if s_eff == 1:
                        d2 = later[(z >> 15) % len(later)]
                        hs = [refs[d2][j] for j in S]
                    elif s_eff == 2:
                        draws = vec["c_in_family_uv"]
                        dr = draws[(z >> 15) % len(draws)]
                        hs = [_vec_from_uv(fams[j], None, *dr[j]) for j in S]
                    else:
                        draws = vec["d_out_family_uv"]
                        dr = draws[(z >> 15) % len(draws)]
                        hs = [_vec_from_uv(fams[j], None, *dr[j]) for j in S]
                k = len(S)
                pairs_ls = [lsets(j, h) for j, h in zip(S, hs)]
                ls = np.column_stack([a for a, _ in pairs_ls])
                lab = np.column_stack([b for _, b in pairs_ls]).astype(np.int8)
                n_amb = int((ls == LS_BOTH).any(axis=1).sum())
                if stat_kind == "1":
                    gs = [g_count(1, 1)]
                elif stat_kind in ("2", "3"):
                    gs = [g_count(k, pat) for pat in range(1 << k)] + ([g_parity(k)] if stat_kind == "2" else [])
                else:
                    p_list = [ds.positions[j] for j in S]
                    A = [assignment_matrix(p.cal.z0, p.cal.z1, p.cal.frame, h) for p, h in zip(p_list, hs)]
                    gs = [mitigation_g(A, g_parity(k))]
                ivs = intervals_multi(ls, gs, exact=True)
                vals = exact_values_multi(lab, gs, exact=True)
                for g, (lo, hi), val in zip(gs, ivs, vals):
                    osc = max(float(np.max(g) - np.min(g)), 0.0)
                    stats["statistics"] += 1
                    if not (lo <= val <= hi):
                        stats["violations"].append({"class": c, "S": S, "lo": str(lo), "val": str(val), "hi": str(hi)})
                    if hi - lo > Fraction(osc) * n_amb:
                        stats["width_bound_violations"].append({"class": c, "S": S, "width": str(hi - lo),
                                                                "bound": osc * n_amb})
                stats["pairs"] += 1
                stats["by_class"][str(c)] = stats["by_class"].get(str(c), 0) + 1
            stats |= {"freeze_sha256": sha, "dataset": ds.name, "family": fname, "later_epochs": later,
                      "seconds": round(time.time() - t0, 1), "n_violations": len(stats["violations"]),
                      "n_width_bound_violations": len(stats["width_bound_violations"])}
            f.write_text(json.dumps(stats, indent=1), encoding="utf-8")
            print(ds.name, fname, stats["pairs"], "pairs", stats["statistics"], "statistics; violations",
                  stats["n_violations"], "width", stats["n_width_bound_violations"], stats["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["q1", "he"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    sh = tuple(int(v) for v in a.shard.split("/"))
    q1(sh) if a.part == "q1" else he(sh)


if __name__ == "__main__":
    main()
