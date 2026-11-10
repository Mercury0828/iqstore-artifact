"""M2 DEV selections for prereg/M2_dev_freeze.json (prereg/M2.md v4 §2, §3; AMENDMENT_C1). DEV epochs 1-5 only.

Parts (each writes runs/m2_dev/<part>.json; resumable per dataset):
  fdrift   F-drift widths (prereg/M2.md §2): 99th percentile over physical positions and consecutive DEV-epoch pairs of
           |re-fitted reference of epoch e+1, in epoch e's frame - reference of epoch e| (dtheta wrapped to (-pi, pi]),
           x 1.5; one pair for final, one for mid-circuit positions. Also the prereg/M1.md §5 planning-threshold check.
  iqgrid   IQSTORE DEV-frozen settings: d0 in {2,3,4} x max depth in {12,16,20} x split fraction in {0.25,0.5,1.0};
           b-bar (container bytes, everything included) per DEV dataset at tau = 1e-3 under F-boot; the selected setting
           has the smallest median over DEV datasets, ties to the smaller grid index (grid order as listed).

Usage: python eval/m2_dev.py fdrift | iqgrid
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import m2_data as M  # noqa: E402
from iqstore.family import fit_lda  # noqa: E402
from iqstore.ingest import ingest_position  # noqa: E402
from iqstore.storage import bits_per_shot_position, write_container  # noqa: E402

OUT = ROOT / "runs" / "m2_dev"  # --out overrides (r1 re-runs: runs/m2_dev/r1)
FDRIFT = ROOT / "runs" / "m2_dev" / "fdrift.json"  # F-drift widths do not depend on the storage format
TWO_PI = 2 * np.pi
TAU = 1e-3
IQ_GRID = list(itertools.product((2, 3, 4), (12, 16, 20), (0.25, 0.5, 1.0)))


def _wrap(a: float) -> float:
    return float((a + np.pi) % TWO_PI - np.pi)


def fdrift() -> dict:
    eps = {e: M.load_epoch(e) for e in M.DEV_EPOCHS}
    obs = {"final": [], "mid": []}
    for e in M.DEV_EPOCHS[:-1]:
        a, b = eps[e], eps[e + 1]
        for (q, ctx), ca in sorted(a.cal.items()):
            cb = b.cal[(q, ctx)]
            x0, y0 = ca.frame.apply(cb.z0)
            x1, y1 = ca.frame.apply(cb.z1)
            h = fit_lda(x0, y0, x1, y1)
            obs[ctx].append({"pair": [e, e + 1], "qubit": q, "dtheta": _wrap(h.theta - ca.ref.theta),
                             "dbeta": float(h.beta - ca.ref.beta)})
    res = {"rule": "1.5 x q99 over positions and consecutive DEV-epoch pairs (prereg/M2.md section 2)", "contexts": {}}
    for ctx, rows in obs.items():
        dt = np.abs([r["dtheta"] for r in rows])
        db = np.abs([r["dbeta"] for r in rows])
        q99t, q99b = float(np.quantile(dt, 0.99)), float(np.quantile(db, 0.99))
        per_pos = {}
        for r in rows:
            per_pos.setdefault(r["qubit"], []).append(abs(r["dtheta"]) > 0.1 or abs(r["dbeta"]) > 0.2)
        res["contexts"][ctx] = {
            "n_observations": len(rows), "q99_abs_dtheta": q99t, "q99_abs_dbeta": q99b,
            "dtheta_half_width": 1.5 * q99t, "dbeta_half_width": 1.5 * q99b,
            "planning_threshold_check": {"q99_dtheta_le_0.1": q99t <= 0.1, "q99_dbeta_le_0.2": q99b <= 0.2,
                                         "positions_with_any_exceedance": sum(any(v) for v in per_pos.values()),
                                         "n_positions": len(per_pos)},
            "observations": rows}
    return res


def _grouped(blobs: list[bytes]) -> list[bytes]:
    return [len(blobs).to_bytes(2, "little")] + blobs


def provenance() -> dict:
    """Code commit (must be clean for iqstore/, baselines/, eval/), environment fingerprint and DEV job ids, written into
    every DEV selection output; resume and the freeze accept only outputs with the same provenance."""
    import importlib.metadata as md
    import platform
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "iqstore", "baselines", "eval"], cwd=ROOT).returncode
    pk = {}
    for name in ("numpy", "numba", "zstandard", "duckdb", "zfpy", "pysz", "pyarrow"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    return {"code_commit": head + ("+dirty" if dirty else ""), "python": platform.python_version(), "packages": pk,
            "dev_jobs": {str(e): j["job_id"] for e, j in M.ledger_jobs().items() if e in M.DEV_EPOCHS}}


def _check_prov(stored: dict | None, cur: dict, what: str) -> None:
    if stored != cur:
        raise RuntimeError(f"{what}: provenance differs from the current code/environment; use a fresh --out")


def _grid_done() -> dict:
    D = {}
    cur = provenance()
    for f in sorted(OUT.glob("iqgrid*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        _check_prov(r.get("provenance"), cur, f.name)
        for n, row in r["datasets"].items():
            if n in D and D[n] != row:
                raise RuntimeError(f"iqgrid: {n} twice with different content")
            D[n] = row
    return D


def iqgrid(shard: tuple[int, int] = (0, 1)) -> dict:
    k_sh, n_sh = shard
    path = OUT / ("iqgrid.json" if n_sh == 1 else f"iqgrid_s{k_sh}of{n_sh}.json")
    cur = provenance()
    if "+dirty" in cur["code_commit"]:
        raise RuntimeError("commit the code first (provenance)")
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"grid": IQ_GRID, "tau": TAU,
                                                                              "family": "F-boot", "datasets": {},
                                                                              "provenance": cur}
    _check_prov(res.get("provenance"), cur, path.name)
    done = set(_grid_done())
    store = OUT / "containers"
    for i_ds, (ep, ds) in enumerate(M.dev_datasets()):
        if ds.name in res["datasets"] or ds.name in done or i_ds % n_sh != k_sh:
            continue
        t0 = time.time()
        N, m = ds.positions[0].raw.size, len(ds.positions)
        xy = [p.cal.frame.apply(p.raw) for p in ds.positions]
        row = {"epoch": ds.epoch, "N": N, "m": m, "bits": [], "exceptions": [], "seconds": []}
        for gi, (d0, dmax, sf) in enumerate(IQ_GRID):
            t1 = time.time()
            blobs, nexc = [], 0
            for p, (x, y) in zip(ds.positions, xy):
                fam = p.fams["F-boot"]
                rep, ib, info = ingest_position(p.cal.frame, fam, x, y, p.raw, TAU, d0, dmax, sf)
                blobs += _grouped(ib)
                nexc += int(info["exc"].sum())
            nb = write_container(store / f"{ds.name}_g{gi}.bin", blobs, "IQSTORE", N, m)
            (store / f"{ds.name}_g{gi}.bin").unlink()
            row["bits"].append(bits_per_shot_position(nb, N, m))
            row["exceptions"].append(nexc)
            row["seconds"].append(round(time.time() - t1, 2))
        row["seconds_total"] = round(time.time() - t0, 1)
        res["datasets"][ds.name] = row
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(ds.name, row["seconds_total"], "s; best", int(np.argmin(row["bits"])), round(min(row["bits"]), 4), flush=True)
    if n_sh != 1:
        return res
    res["datasets"] = _grid_done() | res["datasets"]
    if len(res["datasets"]) != sum(1 for e in M.workload()["datasets"] if e["split"] == "DEV"):
        raise RuntimeError("iqgrid: not every DEV dataset is done (run the remaining shards, then `iqgrid` again)")
    med = np.median(np.array([r["bits"] for r in res["datasets"].values()]), axis=0)
    res["median_bits_per_setting"] = med.tolist()
    sel = int(np.flatnonzero(med == med.min())[0])
    res["selected"] = {"grid_index": sel, "d0": IQ_GRID[sel][0], "max_depth": IQ_GRID[sel][1],
                       "split_fraction": IQ_GRID[sel][2], "median_bits": float(med[sel])}
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def fdrift_family(p, fd: dict):
    from iqstore.family import family_from_widths
    c = fd["contexts"][p.context]
    return family_from_widths(p.cal.ref, c["dtheta_half_width"], c["dbeta_half_width"])


def storage() -> dict:
    """DEV storage preview (W3 component (a) on DEV; diagnostic only, never a TEST number)."""
    from eval.m2_storage import lossless_row, storage_row
    iq = json.loads((OUT / "iqgrid.json").read_text(encoding="utf-8"))["selected"]
    fd = json.loads(FDRIFT.read_text(encoding="utf-8"))
    path = OUT / "storage.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"iqstore_settings": iq, "datasets": {}}
    store = OUT / "containers"
    for ep, ds in M.dev_datasets():
        if ds.name in res["datasets"]:
            continue
        t0 = time.time()
        row = {"epoch": ds.epoch, "m": len(ds.positions)} | lossless_row(ds, store)
        for fname in ("F-boot", "F-plan", "F-drift"):
            fams = [fdrift_family(p, fd) for p in ds.positions] if fname == "F-drift" else None
            for tau in (1e-2, 1e-3, 1e-4):
                # B2 (136 x 2 eps candidates per position) is not needed for the DEV diagnostic; it runs on TEST
                row[f"{fname}|{tau:g}"] = storage_row(ds, fname, tau, iq, store, methods=("IQSTORE", "B4q", "B4"),
                                                      families=fams)
        res["datasets"][ds.name] = row
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        r = row["F-boot|0.001"]
        print(ds.name, round(time.time() - t0, 1), "s; F-boot 1e-3 IQ", round(r["IQSTORE"], 4), "B4q",
              round(r["B4q"], 4), "ratio", round(r["B4q"] / r["IQSTORE"], 4), flush=True)
    return res


def b3a6() -> dict:
    """DEV-frozen settings of B3 (bins in {64^2, 128^2, 256^2}) and A6 (block size in {256, 1024, 4096}): smallest median
    b-bar over DEV datasets (prereg/M2.md section 3; neither depends on tau or the family), ties to the first."""
    from baselines.storage_baselines import a6_blobs, b3_blobs
    from eval.m2_storage import _grouped

    grids = {"B3": (64, 128, 256), "A6": (256, 1024, 4096)}
    res = {m_: {"grid": list(g), "datasets": {}} for m_, g in grids.items()} | {"provenance": provenance()}
    store = OUT / "containers"
    store.mkdir(parents=True, exist_ok=True)
    for ep, ds in M.dev_datasets(families=False):
        N, m = ds.positions[0].raw.size, len(ds.positions)
        a6c = [dict() for _ in ds.positions]
        for meth, g in grids.items():
            bits = []
            for v in g:
                bl = []
                for jj, p in enumerate(ds.positions):
                    if meth == "B3":
                        x, y = p.cal.frame.apply(p.raw)
                        bl += _grouped(b3_blobs(x, y, p.cal.frame, v))
                    else:
                        bl += _grouped(a6_blobs(p.raw, p.cal.frame, p.cal.ref, v, a6c[jj]))
                f = store / f"{meth}_{ds.name}_{v}.bin"
                nb = write_container(f, bl, meth, N, m)
                f.unlink()
                bits.append(bits_per_shot_position(nb, N, m))
            res[meth]["datasets"][ds.name] = bits
        print(ds.name, {k: res[k]["datasets"][ds.name] for k in grids}, flush=True)
    for meth, g in grids.items():
        med = np.median(np.array(list(res[meth]["datasets"].values())), axis=0)
        sel = int(np.flatnonzero(med == med.min())[0])
        res[meth]["median_bits"] = med.tolist()
        res[meth]["selected"] = g[sel]
    (OUT / "b3a6.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def b0q() -> dict:
    """B0q engine selection (prereg/M2.md §3 [F9], §6 timing protocol; sampling per amendment A15).
    Candidates: numpy+numba (`B0File`, from the persisted B0 container) and DuckDB (`DuckFile`, from a persisted DuckDB
    database file), END-TO-END: every run opens the file by path and decodes what the query needs; nothing decoded is
    retained between runs (OS cache warm, process alive, JIT compiled). Threads = 1 everywhere (checked). Both engines
    are exact; a mismatch with the exact reference disqualifies an engine.
    DEV query mix (A15): per DEV dataset and query class 1-5, 20 (instance, vector) pairs drawn with STREAM 8; classes
    1-4 with vector sources (a) reference / (b) later DEV refit / (c) in-family / (d) out-of-family (F-drift family,
    normalized coordinates of M2_workload.json); class 5 = a pair at a sweep vector; class 4 completed to the mitigated
    value <g, counts> (g = M^T f built outside the timed interval, identically for both engines).
    Runs: per (engine, pair) 2 warm-ups, then 10 measured runs; within each phase the engine order is shuffled
    with STREAM 8; the pair's time = median of its 10 measured runs. Primary = the eligible engine with the smaller
    median over all pairs (equal count per class)."""
    import os
    from baselines.storage_baselines import b0_blobs
    from eval.b0q import B0File, B0FileSel, DuckFile

    ENGINES = ("numba", "numba_sel", "duckdb")  # amendment A17 adds the selective reader
    from eval.m2_cov import class4_g, later_refs, queries
    from eval.workload import rng
    from iqstore.family import Discriminator, family_from_widths
    from fractions import Fraction
    from iqstore.query import exact_labels, exact_value, to_float

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise RuntimeError(f"{v} must be 1 (prereg section 6)")
    Q_PER_CLASS = 20
    cur = provenance()
    if "+dirty" in cur["code_commit"]:
        raise RuntimeError("commit the code first (provenance)")
    fd = json.loads(FDRIFT.read_text(encoding="utf-8"))["contexts"]
    w = {e["id"]: e for e in M.workload()["datasets"]}
    eps = {e: M.load_epoch(e) for e in M.DEV_EPOCHS}
    fdir = OUT / "b0q_files"
    fdir.mkdir(parents=True, exist_ok=True)
    res = {"provenance": cur, "protocol": b0q.__doc__, "per_dataset": {}, "times": {e: [] for e in ENGINES},
           "warmups": {e: [] for e in ENGINES}, "mismatches": {e: 0 for e in ENGINES}}
    for ep, ds in M.dev_datasets():
        g = rng("timing_order", ds.index, 0)
        qs = queries(w[ds.name], ds, ep)
        refs = later_refs(eps, ds)
        frames = [p.cal.frame for p in ds.positions]
        f0 = fdir / f"{ds.name}.b0"
        write_container(f0, b0_blobs(ds.raw, frames, [p.fams["F-boot"] for p in ds.positions]), "B0",
                        ds.positions[0].raw.size, len(ds.positions))
        f1 = fdir / f"{ds.name}.duckdb"
        if f1.exists():
            f1.unlink()
        DuckFile.write(f1, ds.raw, frames)
        eng = {"numba": B0File(f0), "numba_sel": B0FileSel(f0), "duckdb": DuckFile(f1)}
        fams = [family_from_widths(p.cal.ref, fd[p.context]["dtheta_half_width"], fd[p.context]["dbeta_half_width"])
                for p in ds.positions]

        def uv(j, u, v):
            f = fams[j]
            return Discriminator(f.th_lo + u * (f.th_hi - f.th_lo), f.b_lo + v * (f.b_hi - f.b_lo))

        times = {e: [] for e in ENGINES}
        for cls in ("1", "2", "3", "4", "5"):
            insts = qs["2" if cls == "5" else cls]
            if not insts:
                continue
            for _ in range(Q_PER_CLASS):
                S = insts[int(g.integers(len(insts)))]
                if cls == "5":
                    u = w[ds.name]["queries"]["5_sweep"]["u"][int(g.integers(5))]
                    v = w[ds.name]["queries"]["5_sweep"]["v"][int(g.integers(10))]
                    hs = [uv(j, u, v) for j in S]
                else:
                    src = int(g.integers(4))
                    if src == 1 and refs:
                        e2 = sorted(refs)[int(g.integers(len(refs)))]
                        hs = [refs[e2][j] for j in S]
                    elif src in (1, 2):
                        dr = w[ds.name]["vectors"]["c_in_family_uv"][int(g.integers(20))]
                        hs = [uv(j, *dr[j]) for j in S]
                    elif src == 3:
                        dr = w[ds.name]["vectors"]["d_out_family_uv"][int(g.integers(10))]
                        hs = [uv(j, *dr[j]) for j in S]
                    else:
                        hs = [ds.positions[j].cal.ref for j in S]
                k = len(S)
                gv = class4_g(ds, S, {j: h for j, h in zip(S, hs)}) if cls == "4" else None
                lab = exact_labels([ds.positions[j].raw for j in S], [frames[j] for j in S], hs)
                ref = np.bincount((lab.astype(np.int64) << np.arange(k)).sum(1), minlength=1 << k)
                ref_val = exact_value(lab, gv) if gv is not None else None  # correctly rounded exact answer

                def run1(name, S=S, hs=hs, gv=gv):
                    c = eng[name].counts(S, hs)
                    if gv is None:
                        return c, None
                    # exact completion <g, counts> (rational accumulation, correctly rounded)
                    v = sum((int(n) * Fraction(float(gv[i])) for i, n in enumerate(c.tolist()) if n), Fraction(0))
                    return c, to_float(v)

                for name in eng:
                    c, v = run1(name)
                    if not np.array_equal(c, ref) or (gv is not None and v != ref_val):
                        res["mismatches"][name] += 1
                tt = {name: [] for name in eng}
                wu = {name: [] for name in eng}
                for phase, reps, store in (("warm-up", 2, wu), ("measured", 10, tt)):
                    seq = [name for name in eng for _ in range(reps)]
                    for o in g.permutation(len(seq)):  # warm-ups strictly before measured runs; order shuffled per phase
                        name = seq[o]
                        t1 = time.perf_counter()
                        run1(name)
                        store[name].append(time.perf_counter() - t1)
                for name in eng:
                    res["warmups"][name] += wu[name]
                    times[name].append(float(np.median(tt[name])))
        res["per_dataset"][ds.name] = {k: float(np.median(v)) for k, v in times.items()} | {"pairs": len(times["numba"])}
        for k in times:
            res["times"][k] += times[k]
        print(ds.name, res["per_dataset"][ds.name], flush=True)
    med = {k: float(np.median(v)) for k, v in res["times"].items()}
    res["median_s"] = med
    res["eligible"] = {k: res["mismatches"][k] == 0 for k in med}
    if not any(res["eligible"].values()):
        raise RuntimeError("no exact B0q engine")
    res["primary"] = min((k for k in med if res["eligible"][k]), key=med.get)
    res["datasets"] = sorted(res["per_dataset"])
    (OUT / "b0q.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def entropy() -> dict:
    """prereg §5 Q6: DEV tertile boundaries of per-position label entropy (binary entropy of the reference-label
    frequency of each DEV dataset position, epochs 1-5), frozen for the TEST label-entropy sweep."""
    from iqstore.query import exact_labels
    H = []
    per = {}
    for ep, ds in M.dev_datasets(families=False):
        lab = exact_labels([p.raw for p in ds.positions], [p.cal.frame for p in ds.positions],
                           [p.cal.ref for p in ds.positions])
        f = lab.mean(axis=0)
        h = [float(-(a * np.log2(a) + (1 - a) * np.log2(1 - a))) if 0 < a < 1 else 0.0 for a in f]
        per[ds.name] = h
        H += h
    q = np.quantile(np.array(H), [1 / 3, 2 / 3])
    res = {"provenance": provenance(), "datasets": sorted(per),
           "rule": "numpy.quantile (linear) at 1/3 and 2/3 over all DEV positions", "n_positions": len(H),
           "tertile_boundaries_bits": [float(q[0]), float(q[1])], "per_dataset": per}
    (OUT / "entropy.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["fdrift", "iqgrid", "storage", "b0q", "b3a6", "entropy"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = pathlib.Path(a.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "fdrift":
        r = fdrift()
        FDRIFT.write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps({c: {k: v for k, v in d.items() if k != "observations"} for c, d in r["contexts"].items()},
                         indent=1))
    elif a.part == "iqgrid":
        r = iqgrid(tuple(int(v) for v in a.shard.split("/")))
        print(json.dumps(r.get("selected", "shard done")))
    elif a.part == "entropy":
        print(json.dumps(entropy()["tertile_boundaries_bits"]))
    elif a.part == "b3a6":
        r = b3a6()
        print(json.dumps({k: (r[k]["selected"], r[k]["median_bits"]) for k in ("B3", "A6")}))
    elif a.part == "b0q":
        r = b0q()
        print(json.dumps({k: r[k] for k in ("median_s", "primary", "mismatches")}))
    else:
        storage()


if __name__ == "__main__":
    main()
