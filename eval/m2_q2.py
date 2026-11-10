"""M2 Q2 timing harness (prereg/M2.md v4 §5 Q2; §6 timing protocol; amendments A12, A13, A15, A17).

  dev-c1-build [--shard k/n]   DEV only, untimed: per DEV dataset the persisted containers of the competitor grid
                               configurations (B4q at width multipliers MULTS of base family F-drift and F-boot, each
                               certified at tau = 1e-3, ingest-time step search as prereg §3), IQSTORE at the frozen
                               settings (bases F-drift / F-boot, multiplier 1 = equal certificate) and the B0 container.
  test-build [--shard k/n]     after the committed freeze, untimed: the same containers for every TEST dataset
                               (runs/m2_test/q2/containers), input of the TEST timing step.
  dev-c1-time                  DEV only, timed (CPU lock held, threads = 1 checked): the DEV query-mix sample of A15
                               (per dataset and class 1-5, 20 (instance, vector) pairs from STREAM 8, extra = 1), every
                               engine end-to-end from its file (prereg §6: opened by path, nothing decoded retained),
                               2 warm-ups + 10 measured runs per (engine, pair), engine order shuffled per phase (STREAM
                               8). Every answer is checked: B0q equal to the exact counts; IQSTORE and B4q intervals
                               contain them. Per configuration: per dataset the mean over classes of the class median
                               (equal weight), then the median over DEV datasets. Writes the A12 product
                               prereg/M2_dev_freeze_c1.json (B4q ordering per base, fastest first, ties to the narrower
                               width) and the evidence runs/m2_dev/c1/c1_times.json. Refuses to overwrite either.

  test-build-verify [--shard]  untimed: rebuild the TEST containers from the clean commit into runs/m2_test/q2/verify
                               and compare byte for byte (provenance of containers built from a dirty tree).
  test-plan [--shard k/n]      after test-build, untimed (amendment A18): per TEST dataset and base family the full
                               prereg section 4 workload (`full_pairs`), exact answers, containment of every interval
                               configuration, realized normalized widths and the A12 matched-width B4q configuration
                               (runs/m2_test/q2/plan).
  test-time [--dataset name]   timed, CPU lock or quiet window held (A18): every plan pair, engines IQSTORE x1, B4q
                               matched, B4q x1, B0q; prereg section 6 protocol (runs/m2_test/q2/time). Warm only; the
                               cold claim needs an approved cache-emptying method (prereg section 6).

The A12 product is used only by the (c1) / Q2 matched-width timing on TEST; TEST timing code refuses to run without it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import m2_data as M  # noqa: E402
from eval.workload import rng  # noqa: E402

C1 = ROOT / "runs" / "m2_dev" / "c1"
C1_FREEZE = ROOT / "prereg" / "M2_dev_freeze_c1.json"
Q2 = ROOT / "runs" / "m2_test" / "q2"
TAU = 1e-3
CLASSES = ("1", "2", "3", "4", "5")
PAIRS_PER_CLASS = 20


def _frozen() -> tuple[dict, str]:
    from eval.m2_test import freeze
    return freeze()


def _code_commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "iqstore", "baselines", "eval"], cwd=ROOT).returncode
    return head + ("+dirty" if dirty else "")


def _grouped(blobs: list[bytes]) -> list[bytes]:
    return [len(blobs).to_bytes(2, "little")] + blobs


def _configs() -> list[tuple[str, str, float]]:
    from eval.m2_cov import BASES, MULTS
    out = [("IQSTORE", b, 1.0) for b in BASES]
    out += [("B4q", b, float(mu)) for b in BASES for mu in MULTS]
    return out


def _fname(meth: str, base: str, mult: float) -> str:
    return f"{meth}_{base}_x{mult:g}.bin"


def dev_build(shard: tuple[int, int] = (0, 1), split: str = "DEV", croot: pathlib.Path | None = None) -> None:
    """Untimed: the containers of every configuration (resumable per dataset). split "DEV": DEV datasets into
    runs/m2_dev/c1/containers; split "TEST" (after the committed freeze; TEST read through eval/m2_test.test_datasets):
    TEST datasets into runs/m2_test/q2/containers, input of the TEST timing step. Same code and settings for both."""
    from baselines.storage_baselines import b0_blobs, b4_slab, b4q_position, fixed_margins
    from eval.m2_cov import base_half
    from eval.m2_storage import DELTAS
    from iqstore.family import family_from_widths
    from iqstore.ingest import ingest_position
    from iqstore.storage import write_container

    fz, fsha = _frozen()
    iq = fz["IQSTORE"]
    fd = {"contexts": fz["F-drift"]}
    k_sh, n_sh = shard
    if split == "DEV":
        items = M.dev_datasets()
        croot = C1 / "containers"
    else:
        from eval.m2_test import test_datasets
        items = ((ep, ds) for _, ep, ds in test_datasets())
        croot = croot or Q2 / "containers"
    for i_ds, (ep, ds) in enumerate(items):
        if i_ds % n_sh != k_sh:
            continue
        d = croot / ds.name
        done = d / "DONE.json"
        if done.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        N, m = ds.positions[0].raw.size, len(ds.positions)
        frames = [p.cal.frame for p in ds.positions]
        write_container(d / "B0.bin", b0_blobs(ds.raw, frames, [p.fams["F-boot"] for p in ds.positions]), "B0", N, m)
        xy = [p.cal.frame.apply(p.raw) for p in ds.positions]
        info = {}
        for meth, base, mult in _configs():
            blobs = []
            fb = 0
            for p, (x, y) in zip(ds.positions, xy):
                hw = base_half(p, base, fd)
                fam = family_from_widths(p.cal.ref, mult * hw[0], mult * hw[1])
                if meth == "IQSTORE":
                    _, ib, _ = ingest_position(p.cal.frame, fam, x, y, p.raw, TAU, iq["d0"], iq["max_depth"],
                                               iq["split_fraction"])
                    blobs += _grouped(ib)
                else:
                    box = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
                    M0, M1 = fixed_margins(p.cal.ref, fam, box)
                    in_slab, side1 = b4_slab(x, y, p.cal.ref, M0, M1)
                    qb, qinfo = b4q_position(x, y, in_slab, side1, p.raw, p.cal.frame, p.cal.ref, M0, M1, fam, TAU,
                                             DELTAS)
                    fb += int(qinfo["mode"] != "grid")
                    blobs += _grouped(qb)
            nbytes = write_container(d / _fname(meth, base, mult), blobs, meth, N, m)
            info[_fname(meth, base, mult)] = {"bytes": nbytes, "fallbacks": fb}
        done.write_text(json.dumps({"freeze_sha256": fsha, "code_commit": _code_commit(), "seconds": time.time() - t0,
                                    "files": info}, indent=1), encoding="utf-8")
        print(ds.name, f"{time.time() - t0:.0f} s", flush=True)


def _pairs(ds, entry, ep, refs, fams, g):
    """The A15 sample: per class 1-5, PAIRS_PER_CLASS (instance, vector) pairs; same procedure as m2_dev.b0q."""
    from eval.m2_cov import class4_g, queries
    from iqstore.family import Discriminator

    qs = queries(entry, ds, ep)

    def uv(j, u, v):
        f = fams[j]
        return Discriminator(f.th_lo + u * (f.th_hi - f.th_lo), f.b_lo + v * (f.b_hi - f.b_lo))

    out = []
    for cls in CLASSES:
        insts = qs["2" if cls == "5" else cls]
        if not insts:
            continue
        for _ in range(PAIRS_PER_CLASS):
            S = insts[int(g.integers(len(insts)))]
            if cls == "5":
                u = entry["queries"]["5_sweep"]["u"][int(g.integers(5))]
                v = entry["queries"]["5_sweep"]["v"][int(g.integers(10))]
                hs = [uv(j, u, v) for j in S]
            else:
                src = int(g.integers(4))
                if src == 1 and refs:
                    e2 = sorted(refs)[int(g.integers(len(refs)))]
                    hs = [refs[e2][j] for j in S]
                elif src in (1, 2):
                    dr = entry["vectors"]["c_in_family_uv"][int(g.integers(20))]
                    hs = [uv(j, *dr[j]) for j in S]
                elif src == 3:
                    dr = entry["vectors"]["d_out_family_uv"][int(g.integers(10))]
                    hs = [uv(j, *dr[j]) for j in S]
                else:
                    hs = [ds.positions[j].cal.ref for j in S]
            gv = class4_g(ds, S, {j: h for j, h in zip(S, hs)}) if cls == "4" else None
            out.append((cls, list(S), hs, gv))
    return out


def dev_time() -> None:
    from baselines.b4q_query import B4qFile
    from eval.b0q import B0File, B0FileSel
    from eval.m2_cov import later_refs
    from iqstore.e2e import IQFile
    from iqstore.family import family_from_widths
    from iqstore.query import exact_dot, exact_labels, exact_value, interval_from_tuples

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise RuntimeError(f"{v} must be 1 (prereg section 6)")
    if C1_FREEZE.exists() or (C1 / "c1_times.json").exists():
        raise SystemExit("A12 product exists; refusing to overwrite")
    fz, fsha = _frozen()
    commit = _code_commit()
    if "+dirty" in commit:
        raise RuntimeError("commit the code first")
    primary = fz["B0q_engine"]["primary"]
    if primary not in ("numba", "numba_sel"):
        raise RuntimeError(f"B0q primary {primary}: add its file engine here")
    fdc = fz["F-drift"]
    w = {e["id"]: e for e in M.workload()["datasets"]}
    eps = {e: M.load_epoch(e) for e in M.DEV_EPOCHS}
    configs = _configs()
    names = [f"{a}|{b}|{c:g}" for a, b, c in configs] + ["B0q"]
    res = {"protocol": __doc__, "freeze_sha256": fsha, "code_commit": commit, "b0q_primary": primary,
           "per_dataset": {}, "violations": {n: 0 for n in names}}
    for ep, ds in M.dev_datasets():
        d = C1 / "containers" / ds.name
        if not (d / "DONE.json").exists():
            raise RuntimeError(f"{ds.name}: containers not built")
        eng = {}
        for (meth, base, mult), n in zip(configs, names):
            f = d / _fname(meth, base, mult)
            eng[n] = IQFile(f) if meth == "IQSTORE" else B4qFile(f)
        eng["B0q"] = (B0FileSel if primary == "numba_sel" else B0File)(d / "B0.bin")
        fams = [family_from_widths(p.cal.ref, fdc[p.context]["dtheta_half_width"], fdc[p.context]["dbeta_half_width"])
                for p in ds.positions]
        g = rng("timing_order", ds.index, 1)
        pairs = _pairs(ds, w[ds.name], ep, later_refs(eps, ds), fams, g)
        frames = [p.cal.frame for p in ds.positions]
        per = {n: {c: [] for c in CLASSES} for n in names}

        for cls, S, hs, gv in pairs:
            lab = exact_labels([ds.positions[j].raw for j in S], [frames[j] for j in S], hs)
            k = len(S)
            ref = np.bincount((lab.astype(np.int64) << np.arange(k)).sum(1), minlength=1 << k)

            ref_val = exact_value(lab, gv) if gv is not None else None

            def run1(n, S=S, hs=hs, gv=gv):
                # class 4 completed to the mitigated value: B0q <g, counts>, interval engines the T1 interval from
                # their label-set tuples; both exact and correctly rounded (iqstore/query.py)
                if n == "B0q":
                    c = eng[n].counts(S, hs)
                    return c, (exact_dot(c, gv) if gv is not None else None)
                lo, hi, plo, phi, tup = eng[n].query(S, hs)
                return (lo, hi), (interval_from_tuples(tup, len(S), gv) if gv is not None else None)

            for n in names:
                c, v = run1(n)
                if n == "B0q":
                    ok = np.array_equal(c, ref) and (gv is None or v == ref_val)
                else:
                    ok = bool(np.all(c[0] <= ref) and np.all(ref <= c[1])) and (gv is None or v[0] <= ref_val <= v[1])
                if not ok:
                    res["violations"][n] += 1
            tt = {n: [] for n in names}
            for phase, reps in (("warm-up", 2), ("measured", 10)):
                seq = [n for n in names for _ in range(reps)]
                for o in g.permutation(len(seq)):
                    n = seq[o]
                    t1 = time.perf_counter()
                    run1(n)
                    dt = time.perf_counter() - t1
                    if phase == "measured":
                        tt[n].append(dt)
            for n in names:
                per[n][cls].append(float(np.median(tt[n])))
        row = {}
        for n in names:
            cm = {c: float(np.median(v)) for c, v in per[n].items() if v}
            row[n] = {"class_median_s": cm, "mix_s": float(np.mean(list(cm.values())))}
        res["per_dataset"][ds.name] = row
        print(ds.name, {n: f"{row[n]['mix_s'] * 1e6:.0f}us" for n in names}, flush=True)
    if any(res["violations"].values()):
        raise RuntimeError(f"answer check failed: {res['violations']}")
    med = {n: float(np.median([res["per_dataset"][x][n]["mix_s"] for x in res["per_dataset"]])) for n in names}
    res["dev_median_mix_s"] = med
    order = {}
    from eval.m2_cov import BASES, MULTS
    for base in BASES:
        cands = [(med[f"B4q|{base}|{float(mu):g}"], float(mu)) for mu in MULTS]
        order[f"B4q|{base}"] = [mu for _, mu in sorted(cands)]  # ties: tuples sort by the narrower multiplier
    C1.mkdir(parents=True, exist_ok=True)
    ev = C1 / "c1_times.json"
    ev.write_text(json.dumps(res, indent=1), encoding="utf-8")
    prod = {"amendment": "docs/guide_amendments.md A12", "freeze_sha256": fsha, "code_commit": commit,
            "engines": {"IQSTORE": "iqstore/lean.py IQFile", "B4q": "baselines/b4q_query.py B4qFile",
                        "B0q": f"eval/b0q.py ({primary})"},
            "protocol": "A15 sample of the DEV query mix (20 pairs per class and dataset, STREAM 8 extra 1); end-to-end "
                        "from the file; 2 warm-ups + 10 measured runs per (engine, pair), shuffled per phase; per "
                        "configuration: mean over classes of the class median, median over DEV datasets",
            "dev_median_mix_s": med, "ordering_fastest_first": order,
            "tie_rule": "equal DEV median: the narrower width (smaller multiplier) first",
            "evidence": {ev.relative_to(ROOT).as_posix(): hashlib.sha256(ev.read_bytes()).hexdigest()}}
    C1_FREEZE.write_text(json.dumps(prod, indent=1), encoding="utf-8")
    print("written", C1_FREEZE, hashlib.sha256(C1_FREEZE.read_bytes()).hexdigest())


# ---------------------------------------------------------------------------------------------------- TEST (A18)

SRC = {"a": 0, "b": 1, "c": 2, "d": 3, "sweep": 4}
N_C, N_D, N_SWEEP = 20, 10, 50


def _test_families(ds, fz, base):
    """The base family of each position, as in eval/m2_test.he (vector sources (c), (d) and the sweep use it)."""
    from eval.m2_test import fdrift_families
    return fdrift_families(ds, fz) if base == "F-drift" else [p.fams["F-boot"] for p in ds.positions]


def full_pairs(entry, ds, ep, refs, fams):
    """A18: the complete prereg section 4 query workload of one TEST dataset, in canonical order. Classes 1-4: every
    instance of the class x every vector: (a) the ingest-day reference, (b) the re-fit of every accepted later epoch,
    ascending, (c) the 20 in-family draws, (d) the 10 out-of-family draws (M2_workload.json). Class 5: every class-2
    instance x the 50 sweep vectors (u-major). Yields (cls, S, src, idx, hs)."""
    from eval.m2_cov import queries
    from eval.m2_test import _vec_from_uv

    qs = queries(entry, ds, ep)
    vec = entry["vectors"]
    later = sorted(refs)
    for cls in ("1", "2", "3", "4"):
        for S in qs[cls]:
            S = list(S)
            yield cls, S, SRC["a"], 0, [ds.positions[j].cal.ref for j in S]
            for e2 in later:
                yield cls, S, SRC["b"], e2, [refs[e2][j] for j in S]
            for i, dr in enumerate(vec["c_in_family_uv"]):
                yield cls, S, SRC["c"], i, [_vec_from_uv(fams[j], None, *dr[j]) for j in S]
            for i, dr in enumerate(vec["d_out_family_uv"]):
                yield cls, S, SRC["d"], i, [_vec_from_uv(fams[j], None, *dr[j]) for j in S]
    sw = entry["queries"]["5_sweep"]
    for S in qs["2"]:
        S = list(S)
        for i, u in enumerate(sw["u"]):
            for jv, v in enumerate(sw["v"]):
                yield "5", S, SRC["sweep"], i * len(sw["v"]) + jv, [_vec_from_uv(fams[j], None, u, v) for j in S]


def _parity(counts) -> int:
    """sum_p (-1)^popcount(p) counts[p]: the integer parity sum of <Z_S> (times N); identical work for every engine."""
    return int(sum(-c if bin(p).count("1") & 1 else c for p, c in enumerate(np.asarray(counts).tolist())))


def _n_amb(tup) -> int:
    """Shots with at least one ambiguous position: tuple keys with some base-4 digit = 3 (iqstore/query.py codes)."""
    tot = 0
    for key in np.flatnonzero(tup).tolist():
        v = key
        while v:
            if v % 4 == 3:
                tot += int(tup[key])
                break
            v //= 4
    return tot


def norm_width(cls: str, out, k: int, N: int, gv, iv4=None):
    """A18: realized normalized width of an interval answer (exact Fraction), scaled so that the single-position
    certificate bound of every class is |S| tau (prereg Q4 useful-query rule): counts and marginals max_p (F+ - F-) / N;
    class 2 / 5 additionally the parity <Z_S> width / (2 N); class 4 the exact mitigated width / (N osc(g)). iv4: the
    exact class-4 interval if already computed."""
    from fractions import Fraction
    from iqstore.query import interval_from_tuples
    lo, hi, plo, phi, tup = out
    if cls == "4":
        a, b = iv4 if iv4 is not None else interval_from_tuples(tup, k, gv, exact=True)
        osc = Fraction(float(np.max(gv))) - Fraction(float(np.min(gv)))
        return (b - a) / (N * osc) if osc > 0 else Fraction(0)
    w = Fraction(int(np.max(np.asarray(hi, dtype=np.int64) - np.asarray(lo, dtype=np.int64))), N)
    if cls in ("2", "5"):
        w = max(w, Fraction(int(phi) - int(plo), 2 * N))
    return w


def check_answer(cls: str, out, ref, k: int, gv, rv) -> tuple[bool, object]:
    """Complete check of one interval-engine answer (exact): pattern-count containment; classes 2 / 5 parity
    containment plo <= parity(ref) <= phi; class 4 exact mitigated containment; T1 width bounds from the ambiguous-shot
    count n (counts <= n, parity <= 2 n, mitigated <= osc(g) n). Returns (ok, exact class-4 interval or None)."""
    from fractions import Fraction
    from iqstore.query import interval_from_tuples
    lo, hi, plo, phi, tup = out
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    n = _n_amb(tup)
    ok = bool(np.all(lo <= ref) and np.all(ref <= hi) and np.all(hi - lo <= n))
    iv = None
    if cls in ("2", "5"):
        par = _parity(ref)
        ok = ok and int(plo) <= par <= int(phi) and int(phi) - int(plo) <= 2 * n
    if cls == "4":
        iv = interval_from_tuples(tup, k, gv, exact=True)
        osc = Fraction(float(np.max(gv))) - Fraction(float(np.min(gv)))
        ok = ok and iv[0] <= rv <= iv[1] and iv[1] - iv[0] <= osc * n
    return ok, iv


def plan_pair(ds, frames, N, b0, iq, b4, order, cls, S, hs, tau=None):
    """One A18 plan pair: exact answer (raw labels, exact rationals; B0q must equal it), complete check of IQSTORE x1
    and every B4q width (check_answer; AssertionError = bug, stop), normalized widths, and the matched configuration:
    the first multiplier in the A12 DEV ordering whose width <= w_IQ + 0.05 |S| tau, else -1 (width-unmatched).
    Returns (counts, exact class-4 value or None, widths, matched)."""
    from fractions import Fraction
    from eval.m2_cov import class4_g
    from iqstore.query import exact_dot, exact_labels, exact_value

    tau = Fraction(1, 1000) if tau is None else tau
    k = len(S)
    lab = exact_labels([ds.positions[j].raw for j in S], [frames[j] for j in S], hs)
    ref = np.bincount((lab.astype(np.int64) << np.arange(k)).sum(1), minlength=1 << k)
    gv = class4_g(ds, S, {j: h for j, h in zip(S, hs)}) if cls == "4" else None
    rv = exact_value(lab, gv, exact=True) if gv is not None else None
    c = b0.counts(S, hs)
    if not np.array_equal(c, ref) or (gv is not None and exact_dot(c, gv, exact=True) != rv):
        raise AssertionError("B0q differs from the exact answer")
    ws = {}
    for name, eng in [("IQ", iq)] + list(b4.items()):
        o = eng.query(S, hs)
        ok, iv = check_answer(cls, o, ref, k, gv, rv)
        if not ok:
            raise AssertionError(f"answer check failed for {name}")
        ws[name] = norm_width(cls, o, k, N, gv, iv)
    lim = ws["IQ"] + Fraction(5, 100) * k * tau
    m = next((mu for mu in order if ws[mu] <= lim), -1.0)
    return ref, rv, ws, m


def _ordering(base: str) -> list[float]:
    c1 = json.loads(C1_FREEZE.read_bytes())
    return [float(x) for x in c1["ordering_fastest_first"][f"B4q|{base}"]]


def _committed_sha(p: pathlib.Path) -> str:
    rel = p.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (p.exists() and tracked and clean):
        raise PermissionError(f"{rel} missing or not committed unmodified")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _c1_sha(fsha: str) -> str:
    sha = _committed_sha(C1_FREEZE)
    if json.loads(C1_FREEZE.read_bytes())["freeze_sha256"] != fsha:
        raise PermissionError("A12 product was made under another DEV freeze")
    return sha


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _files_for(base: str) -> list[str]:
    from eval.m2_cov import MULTS
    return ["B0.bin", _fname("IQSTORE", base, 1.0)] + [_fname("B4q", base, float(mu)) for mu in MULTS]


def _inputs_sha(entry, ds) -> str:
    """Workload entry + the dataset's raw IQ points and calibration (what the plan's exact answers depend on)."""
    h = hashlib.sha256(json.dumps(entry, sort_keys=True).encode())
    h.update(np.ascontiguousarray(ds.raw).tobytes())
    for p in ds.positions:
        h.update(np.ascontiguousarray(p.cal.z0).tobytes() + np.ascontiguousarray(p.cal.z1).tobytes())
    return h.hexdigest()


def _workload_sha(entry, ds, ep, refs, fams) -> str:
    """Every pair of the A18 workload as actually queried: (class, S, source, index) and the exact float64 (theta,
    beta) of every discriminator, plus the base families (bounds) the vectors were mapped through."""
    h = hashlib.sha256()
    for f in fams:
        h.update(np.array([f.th_lo, f.th_hi, f.b_lo, f.b_hi], dtype=np.float64).tobytes())
    for cls, S, src, idx, hs in full_pairs(entry, ds, ep, refs, fams):
        h.update(f"{cls}|{S}|{src}|{idx}|".encode())
        h.update(np.array([(x.theta, x.beta) for x in hs], dtype=np.float64).tobytes())
    return h.hexdigest()


def _code_sha() -> str:
    """The query and planning implementation: every iqstore/*.py, the B4q and B0q engines, the workload and family
    helpers, and the source of the A18 planning functions (a change to any of them invalidates plan and timing)."""
    import inspect
    h = hashlib.sha256()
    files = sorted((ROOT / "iqstore").glob("*.py")) + [ROOT / "baselines" / "b4q_query.py", ROOT / "eval" / "b0q.py",
                                                        ROOT / "eval" / "m2_cov.py", ROOT / "eval" / "m2_test.py",
                                                        ROOT / "eval" / "m2_data.py", ROOT / "eval" / "workload.py"]
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    for fn in (full_pairs, norm_width, check_answer, plan_pair, _parity, _n_amb, _test_families, _ordering, _fname):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def _publish(stem: pathlib.Path, arrays: dict, meta: dict) -> None:
    """Atomic publication of <stem>.npz + <stem>.json (json last; it carries the npz sha256)."""
    tmp = stem.with_name(stem.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    meta["npz_sha256"] = _sha(tmp)
    os.replace(tmp, stem.with_suffix(".npz"))
    tj = stem.with_name(stem.name + ".tmp.json")
    tj.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tj, stem.with_suffix(".json"))


def _complete(stem: pathlib.Path) -> bool:
    """A published artifact pair is complete iff its json names the sha256 of the npz beside it. An npz without a valid
    json (interrupted attempt) is kept under a distinct name (all runs reported) and the step re-runs."""
    j, z = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if j.exists():
        if z.exists() and json.loads(j.read_text(encoding="utf-8")).get("npz_sha256") == _sha(z):
            return True
        raise RuntimeError(f"{j}: npz missing or changed")
    if z.exists():
        os.replace(z, stem.with_name(stem.name + f".interrupted-{int(time.time())}.npz"))
    return False


def test_build_verify(shard: tuple[int, int] = (0, 1)) -> None:
    """Untimed: rebuild every TEST dataset's containers from the CLEAN committed code into runs/m2_test/q2/verify and
    compare every file byte for byte (sha256) with runs/m2_test/q2/containers; writes verify/<dataset>/VERIFY.json.
    Establishes the provenance of containers whose DONE.json records a dirty tree."""
    from eval.m2_test import test_datasets
    if "+dirty" in _code_commit():
        raise RuntimeError("commit the code first")
    _, fsha = _frozen()
    vroot = Q2 / "verify"
    dev_build(shard, split="TEST", croot=vroot)
    for i_ds, (_, _, ds) in enumerate(test_datasets(families=False)):
        if i_ds % shard[1] != shard[0]:
            continue
        a, b = Q2 / "containers" / ds.name, vroot / ds.name
        vd = json.loads((b / "DONE.json").read_text(encoding="utf-8"))
        if "+dirty" in vd["code_commit"] or vd["freeze_sha256"] != fsha:
            raise RuntimeError(f"{ds.name}: the verification rebuild is not from clean code under the active freeze")
        files = sorted(p.name for p in a.glob("*.bin"))
        sa = {fn: _sha(a / fn) for fn in files}
        sb = {p.name: _sha(p) for p in sorted(b.glob("*.bin"))}
        ok = sa == sb
        (b / "VERIFY.json").write_text(json.dumps({"rebuild_code_commit": vd["code_commit"], "freeze_sha256": fsha,
                                                   "containers_sha256": sa, "rebuild_sha256": sb,
                                                   "all_identical": ok}, indent=1), encoding="utf-8")
        print(ds.name, "identical" if ok else "DIFFERENT", flush=True)


def _verified(name: str, fsha: str) -> bool:
    """A clean rebuild under the active freeze reproduced exactly the containers that are there now."""
    v = Q2 / "verify" / name / "VERIFY.json"
    if not v.exists():
        return False
    r = json.loads(v.read_text(encoding="utf-8"))
    if not (r.get("all_identical") and r.get("freeze_sha256") == fsha and "+dirty" not in r.get("rebuild_code_commit",
                                                                                                  "+dirty")):
        return False
    a = Q2 / "containers" / name
    return {p.name: _sha(p) for p in sorted(a.glob("*.bin"))} == r["containers_sha256"]


def test_plan(shard: tuple[int, int] = (0, 1)) -> None:
    """Untimed (co-run allowed): per TEST dataset and base family, the full A18 workload with the exact answer of every
    pair, the complete check of every interval configuration (IQSTORE x1 and the 5 B4q widths; one failure = bug, stop),
    the realized normalized widths and the A12 matched-width B4q configuration of every pair. Binds the plan to the
    freeze, the A12 product, the clean code commit, the container files (sha256) and the inputs. Output
    runs/m2_test/q2/plan/<dataset>_<base>.npz + .json (resumable, atomic)."""
    from baselines.b4q_query import B4qFile
    from eval.b0q import B0FileSel
    from eval.m2_cov import BASES, MULTS, later_refs_all
    from eval.m2_test import test_datasets
    from iqstore.e2e import IQFile

    fz, fsha = _frozen()
    c1sha = _c1_sha(fsha)
    commit = _code_commit()
    if "+dirty" in commit:
        raise RuntimeError("commit the code first")
    csha = _code_sha()
    out = Q2 / "plan"
    out.mkdir(parents=True, exist_ok=True)
    for i_ds, (entry, ep, ds) in enumerate(test_datasets()):
        if i_ds % shard[1] != shard[0]:
            continue
        d = Q2 / "containers" / ds.name
        if not (d / "DONE.json").exists():
            raise RuntimeError(f"{ds.name}: TEST containers not built (test-build)")
        done = json.loads((d / "DONE.json").read_text(encoding="utf-8"))
        if done["freeze_sha256"] != fsha:
            raise RuntimeError(f"{ds.name}: containers built under another freeze")
        if "+dirty" in done["code_commit"] and not _verified(ds.name, fsha):
            raise RuntimeError(f"{ds.name}: containers from a dirty tree and not reproduced (test-build-verify)")
        refs = later_refs_all(ds)
        frames = [p.cal.frame for p in ds.positions]
        N = ds.positions[0].raw.size
        b0 = B0FileSel(d / "B0.bin")
        for base in BASES:
            f = out / f"{ds.name}_{base}"
            fams = _test_families(ds, fz, base)
            if _complete(f):
                pm = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
                if (pm["freeze_sha256"], pm["c1_sha256"], pm["code_sha256"], pm["inputs_sha256"],
                        pm["workload_sha256"]) != (fsha, c1sha, csha, _inputs_sha(entry, ds),
                                                   _workload_sha(entry, ds, ep, refs, fams)):
                    raise RuntimeError(f"{f}: complete plan from another freeze / code / inputs; move it aside")
                if pm["containers_sha256"] != {fn: _sha(d / fn) for fn in _files_for(base)}:
                    raise RuntimeError(f"{f}: complete plan made on other container files; move it aside")
                continue
            t0 = time.time()
            order = _ordering(base)
            iq = IQFile(d / _fname("IQSTORE", base, 1.0))
            b4 = {float(mu): B4qFile(d / _fname("B4q", base, float(mu))) for mu in MULTS}
            rows = {"cls": [], "S": [], "src": [], "idx": [], "w_iq": [], "w_b4q": [], "matched": [], "counts": []}
            c4vals, wex = [], []
            for cls, S, src, idx, hs in full_pairs(entry, ds, ep, refs, fams):
                k = len(S)
                try:
                    ref, rv, ws, m = plan_pair(ds, frames, N, b0, iq, b4, order, cls, S, hs)
                except AssertionError as ex:
                    raise AssertionError(f"{ds.name} {base} class {cls} S {S} src {src}/{idx}: {ex}") from None
                rows["cls"].append(int(cls)); rows["S"].append(S + [-1] * (4 - k)); rows["src"].append(src)
                rows["idx"].append(idx); rows["w_iq"].append(float(ws["IQ"]))
                rows["w_b4q"].append([float(ws[float(mu)]) for mu in MULTS]); rows["matched"].append(m)
                rows["counts"].append(np.pad(ref, (0, 16 - ref.size), constant_values=-1))
                c4vals.append(str(rv) if rv is not None else "")
                if cls == "4":
                    wex.append({"pair": len(rows["cls"]) - 1, "w_iq": str(ws["IQ"]),
                                "w_b4q": {f"{mu:g}": str(ws[float(mu)]) for mu in MULTS}})
            mt = np.array(rows["matched"])
            arrays = {"cls": np.array(rows["cls"], np.int8), "S": np.array(rows["S"], np.int16),
                      "src": np.array(rows["src"], np.int8), "idx": np.array(rows["idx"], np.int16),
                      "w_iq": np.array(rows["w_iq"]), "w_b4q": np.array(rows["w_b4q"]), "matched": mt,
                      "counts": np.array(rows["counts"], np.int32)}
            meta = {"freeze_sha256": fsha, "c1_sha256": c1sha, "code_commit": commit, "dataset": ds.name, "base": base,
                    "N": N, "later_epochs": sorted(refs), "pairs": len(mt), "mults": list(MULTS), "ordering": order,
                    "inputs_sha256": _inputs_sha(entry, ds), "code_sha256": csha,
                    "workload_sha256": _workload_sha(entry, ds, ep, refs, fams),
                    "containers_sha256": {fn: _sha(d / fn) for fn in _files_for(base)},
                    "width_unmatched": int((mt < 0).sum()),
                    "matched_counts": {f"{mu:g}": int((mt == mu).sum()) for mu in MULTS},
                    "class4_values_exact": c4vals, "class4_widths_exact": wex,
                    "seconds": round(time.time() - t0, 1)}
            _publish(f, arrays, meta)
            print(ds.name, base, len(mt), "pairs; unmatched", meta["width_unmatched"], meta["matched_counts"],
                  f"{meta['seconds']} s", flush=True)


def _lock_held() -> bool:
    """The timed block needs an otherwise idle machine. On our workstation this checked a shared lock file; in this
    release it returns True when the environment variable IQSTORE_MACHINE_IDLE is set to 1."""
    import os
    return os.environ.get("IQSTORE_MACHINE_IDLE") == "1"


def _ckpt(ck: pathlib.Path, T, t_op, done: int, meta: dict) -> None:
    """Atomic checkpoint of a timing attempt: all times so far (NaN = not yet run; the non-NaN entries are
    authoritative), done = number of fully timed pairs (pairs 0 .. done-1; pair `done` may hold partial runs), and the
    provenance needed to interpret it (plan json, code and timing-code hashes, engines, dataset / base, attempt)."""
    tmp = ck.with_name(ck.name + ".tmp.npz")
    np.savez_compressed(tmp, t=T, t_operator=t_op, done=np.int64(done), meta=np.array(json.dumps(meta)))
    os.replace(tmp, ck)


def _time_code_sha() -> str:
    """The timing implementation (protocol loop, completed-answer wrappers, checks, publication) on top of _code_sha."""
    import inspect
    h = hashlib.sha256(_code_sha().encode())
    for fn in (test_time, _lock_held, _publish, _complete, _ckpt, check_answer, _parity, _n_amb, rng):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def test_time(dataset: str | None = None) -> None:
    """Timed (CPU lock or quiet window held, re-checked every 500 pairs; threads = 1 checked; single process): for every
    pair of the A18 plan, the engines IQSTORE x1 (equal certificate), B4q at the matched configuration (finest when
    width-unmatched), B4q x1 (equal certificate) and B0q (frozen primary), each end-to-end from its file (prereg section
    6) to the COMPLETED answer: B0q counts plus the parity sum (classes 2 / 5) or the mitigated value <g, counts> (class
    4); interval engines their count / parity intervals plus the mitigated interval from the label-set tuples (class 4);
    g = M^T f is built once per pair outside the engine timing and its construction time is recorded separately
    (identical for every method, prereg section 4 item 4). 2 warm-ups + 10 measured runs per (engine, pair), engine
    order shuffled per phase (STREAM 8, extra 10 + base index). Every answer is checked completely (check_answer) before
    the pair is timed; one failure stops the run. Inputs are bound to the plan (container sha256, inputs sha256, pair
    identities and count). Output runs/m2_test/q2/time/<dataset>_<base>.npz (all 12 run times per engine, warm-ups
    first) + .json (atomic; resumable; an interrupted attempt is kept under a distinct name)."""
    from baselines.b4q_query import B4qFile
    from eval.b0q import B0File, B0FileSel
    from eval.m2_cov import BASES, MULTS, class4_g, later_refs_all
    from eval.m2_test import test_datasets
    from iqstore.e2e import IQFile
    from iqstore.query import exact_dot, interval_from_tuples, to_float

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise RuntimeError(f"{v} must be 1 (prereg section 6)")
    if not _lock_held():
        raise PermissionError("CPU lock / quiet window not held by iqstore")
    fz, fsha = _frozen()
    c1sha = _c1_sha(fsha)
    commit = _code_commit()
    if "+dirty" in commit:
        raise RuntimeError("commit the code first")
    primary = fz["B0q_engine"]["primary"]
    if primary not in ("numba", "numba_sel"):
        raise RuntimeError(f"B0q primary {primary}: add its file engine here")
    csha = _code_sha()
    tsha = _time_code_sha()
    out = Q2 / "time"
    out.mkdir(parents=True, exist_ok=True)
    names = ("IQSTORE", "B4q_matched", "B4q_eq", "B0q")
    for entry, ep, ds in test_datasets():
        if dataset and ds.name != dataset:
            continue
        d = Q2 / "containers" / ds.name
        refs = later_refs_all(ds)
        for ib, base in enumerate(BASES):
            f = out / f"{ds.name}_{base}"
            pf = Q2 / "plan" / f"{ds.name}_{base}"
            if not _complete(pf):
                raise RuntimeError(f"{pf}: no complete plan")
            pjsha = _sha(pf.with_suffix(".json"))
            pm = json.loads(pf.with_suffix(".json").read_text(encoding="utf-8"))
            fams = _test_families(ds, fz, base)
            if pm["freeze_sha256"] != fsha or pm["c1_sha256"] != c1sha or "+dirty" in pm["code_commit"]:
                raise RuntimeError("plan made under another freeze / A12 product, or from uncommitted code")
            if (pm["inputs_sha256"], pm["code_sha256"], pm["workload_sha256"]) != (
                    _inputs_sha(entry, ds), csha, _workload_sha(entry, ds, ep, refs, fams)):
                raise RuntimeError("plan inputs / query code / workload vectors differ")
            for fn, h in pm["containers_sha256"].items():
                if _sha(d / fn) != h:
                    raise RuntimeError(f"{ds.name}/{fn} differs from the planned container")
            if _complete(f):  # only after every plan-context check above
                tm = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
                if (tm["plan_json_sha256"], tm["code_sha256"], tm["time_code_sha256"]) != (pjsha, csha, tsha):
                    raise RuntimeError(f"{f}: complete timing from another plan / code; move it aside")
                continue
            P = np.load(pf.with_suffix(".npz"))
            eng = {"IQSTORE": IQFile(d / _fname("IQSTORE", base, 1.0)),
                   "B0q": (B0FileSel if primary == "numba_sel" else B0File)(d / "B0.bin")}
            b4 = {float(mu): B4qFile(d / _fname("B4q", base, float(mu))) for mu in MULTS}
            finest = min(float(mu) for mu in MULTS)
            g = rng("timing_order", ds.index, 10 + ib)
            n_pairs = len(P["cls"])
            T = np.full((n_pairs, len(names), 12), np.nan)
            t_op = np.full(n_pairs, np.nan)
            t0 = time.time()
            ck = f.with_name(f.name + f".attempt-{int(t0)}.partial.npz")
            ckm = {"dataset": ds.name, "base": base, "attempt": int(t0), "engines": list(names),
                   "plan_json_sha256": pjsha, "code_sha256": csha, "time_code_sha256": tsha, "code_commit": commit}
            i = -1
            try:
                for i, (cls, S, src, idx, hs) in enumerate(full_pairs(entry, ds, ep, refs, fams)):
                    if i and i % 1000 == 0:  # checkpoint of this attempt (kept if the attempt does not complete)
                        _ckpt(ck, T, t_op, i, ckm)
                    k = len(S)
                    if (i >= n_pairs or int(P["cls"][i]) != int(cls) or int(P["src"][i]) != src
                            or int(P["idx"][i]) != idx or P["S"][i].tolist() != S + [-1] * (4 - k)):
                        raise RuntimeError(f"plan / workload mismatch at pair {i}")
                    if i % 500 == 0 and not _lock_held():
                        raise PermissionError("CPU lock / quiet window no longer held")
                    mu = float(P["matched"][i])
                    e = dict(eng, B4q_matched=b4[mu if mu > 0 else finest], B4q_eq=b4[1.0])
                    ref = P["counts"][i][:1 << k].astype(np.int64)
                    gv = None
                    if cls == "4":
                        t1 = time.perf_counter()
                        gv = class4_g(ds, S, {j: h for j, h in zip(S, hs)})
                        t_op[i] = time.perf_counter() - t1
                    rv = None
                    if gv is not None:
                        from fractions import Fraction
                        rv = Fraction(pm["class4_values_exact"][i])

                    def b0q(e=e, S=S, hs=hs, cls=cls, gv=gv):
                        c = e["B0q"].counts(S, hs)
                        if cls in ("2", "5"):
                            return c, _parity(c)
                        if cls == "4":
                            return c, exact_dot(c, gv)
                        return c, None

                    def ivq(n, e=e, S=S, hs=hs, cls=cls, gv=gv, k=k):
                        o = e[n].query(S, hs)
                        return o, (interval_from_tuples(o[4], k, gv) if cls == "4" else None)

                    call = {n: (b0q if n == "B0q" else (lambda n=n: ivq(n))) for n in names}
                    for n in names:  # complete answer check (untimed), before timing this pair
                        if n == "B0q":
                            c, v = call[n]()
                            ok = np.array_equal(c, ref) and (cls not in ("2", "5") or v == _parity(ref)) and \
                                (cls != "4" or (exact_dot(c, gv, exact=True) == rv and v == to_float(rv)))
                        else:
                            o, v = call[n]()
                            ok, iv = check_answer(cls, o, ref, k, gv, rv)
                            if cls == "4":
                                ok = ok and tuple(v) == (to_float(iv[0]), to_float(iv[1]))
                        if not ok:
                            raise RuntimeError(f"{ds.name} {base} pair {i} ({cls} {S}): answer check failed for {n}")
                    for phase, reps, off in (("warm-up", 2, 0), ("measured", 10, 2)):
                        seq = [n for n in names for _ in range(reps)]
                        cnt = {n: 0 for n in names}
                        for o in g.permutation(len(seq)):
                            n = seq[o]
                            fn = call[n]
                            t1 = time.perf_counter()
                            fn()
                            dt = time.perf_counter() - t1
                            T[i, names.index(n), off + cnt[n]] = dt
                            cnt[n] += 1
            except BaseException:
                _ckpt(ck, T, t_op, i, ckm)  # every completed observation of this attempt is kept
                raise
            if i + 1 != n_pairs:
                raise RuntimeError(f"workload has {i + 1} pairs, plan {n_pairs}")
            med = np.median(T[:, :, 2:], axis=2)
            cl = P["cls"]
            matched = P["matched"] > 0
            per_all = {n: {str(c): float(np.median(med[cl == c, j])) for c in range(1, 6) if (cl == c).any()}
                       for j, n in enumerate(names)}
            per_m = {n: {str(c): float(np.median(med[(cl == c) & matched, j]))
                         for c in range(1, 6) if ((cl == c) & matched).any()} for j, n in enumerate(names)}
            meta = {"freeze_sha256": fsha, "c1_sha256": c1sha, "plan_npz_sha256": pm["npz_sha256"],
                    "plan_json_sha256": pjsha, "code_sha256": csha, "time_code_sha256": tsha,
                    "code_commit": commit, "b0q_primary": primary, "dataset": ds.name, "base": base,
                    "engines": list(names), "pairs": int(n_pairs),
                    "pairs_per_class": {str(c): int((cl == c).sum()) for c in range(1, 6)},
                    "matched_pairs_per_class": {str(c): int(((cl == c) & matched).sum()) for c in range(1, 6)},
                    "class_median_s_all_pairs": per_all,
                    "class_median_s_matched_pairs_same_mask": per_m,
                    "note": "all-pair medians include width-unmatched pairs (B4q_matched = finest there); W3 (c1) uses "
                            "the matched-pairs medians, the same mask for IQSTORE and B4q; a class with more than half "
                            "of its pairs unmatched is inconclusive for (c1) (prereg section 5)",
                    "class4_operator_s_median": float(np.nanmedian(t_op)) if np.isfinite(t_op).any() else None,
                    "seconds": round(time.time() - t0, 1)}
            _publish(f, {"t": T, "t_operator": t_op}, meta)
            if ck.exists():
                ck.unlink()  # subsumed by the complete result of the same attempt
            r = {c: per_all["B0q"][c] / per_all["IQSTORE"][c] for c in per_all["IQSTORE"]}
            print(ds.name, base, f"{meta['seconds']} s; B0q/IQ per class (all pairs)",
                  {c: f"{v:.1f}" for c, v in r.items()}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev-c1-build", "dev-c1-time", "test-build", "test-build-verify", "test-plan",
                                    "test-time"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dataset", default=None)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    if a.cmd == "dev-c1-build":
        dev_build((k, n))
    elif a.cmd == "test-build":
        dev_build((k, n), split="TEST")
    elif a.cmd == "test-build-verify":
        test_build_verify((k, n))
    elif a.cmd == "test-plan":
        test_plan((k, n))
    elif a.cmd == "test-time":
        test_time(a.dataset)
    else:
        dev_time()


if __name__ == "__main__":
    main()
