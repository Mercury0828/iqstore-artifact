"""M2 DEV storage-coverage trade-off (prereg/M2.md v4 §5 Q4 "Storage-coverage trade-off", W3 components (b), (c2)).

DEV only (epochs 1-5). Knob grids (prereg §5): family width x {0.5, 1, 1.5, 2, 3} of a base width (F-drift; F-boot as
the second family of the W3 table) for IQSTORE, B4q (+ the sensitivity variant B4q+box) and A7, each certified at
tau = 1e-3; for B2 (zfp, SZ3) the eps grid logspace(-4, 0.5, 136) (a family-independent knob: B2 at eps stores the codec
reconstruction and its measured error; no certificate is imposed on the knob grid).
  * b-bar of every configuration (same container and coder; IQSTORE at the DEV-frozen settings with its ingest search).
  * Useful-query coverage against the later-day vectors (b) inside DEV: for ingest epoch d and later DEV epoch d', the
    LDA refit of epoch d' calibration of the same (qubit, context) in epoch d's frame. Every method answers with its
    exact label-set rule: IQSTORE / A7 band leaves (half-open) and decided regions; B4q cells (half-open) and, outside
    the slab, the stored side bit for in-family vectors and {0, 1} otherwise (its stored bytes do not bound the region);
    B4q+box (+32 B per position) bounds the outside regions by the closed data box (exact rational vertices of
    box and half-plane, widened by the float error of the slab test); B2 closed error squares.
  * Query classes (prereg §4) 1 single counts, 2 adjacent pairs (pattern counts and <ZZ>), 3 the 100 random 3/4-subsets
    (marginals), 4 the mitigated <Z_S> with M from the ingest-day calibration classified by the vector (exact realized
    interval width). An instance is useful iff every statistic satisfies the class bound: counts / marginals
    (F+ - F-)/N <= |S| tau, parity <= 2 |S| tau, mitigated <= |S| tau osc(M^T f).
  * Maintenance work (prereg Q3 / W3 (c2)): the real event stream inside DEV (reference -> each later DEV refit, in
    order, all positions changed), the identical algorithm for IQSTORE and B4q-inc: every indexed cell evaluated under
    old and new h, shots of changed cells read through the cell -> shots index, exceptions re-labelled.
    w = cells / regions evaluated + shot-position entries read or re-labelled (prereg scalar); view entries written are
    reported beside it, not inside w.

Usage: python eval/m2_cov.py run [--out runs/m2_dev/r2] [--shard k/n] | summary [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.a7 import a7_position  # noqa: E402
from baselines.storage_baselines import b2_position, b4_slab, b4q_position, fixed_margins  # noqa: E402
from eval import m2_data as M  # noqa: E402
from eval.m2_storage import DELTAS, EPS_GRID, _grouped, _measure  # noqa: E402
from iqstore.family import Discriminator, family_from_widths, fit_lda  # noqa: E402
from iqstore.ingest import ingest_position  # noqa: E402
from iqstore.query import g_parity, intervals_multi  # noqa: E402
from iqstore.rep import LS_BOTH, LS_ONLY0, LS_ONLY1, closed_rect_label_sets, rect_label_sets  # noqa: E402
from iqstore.views import assignment_matrix, mitigation_g  # noqa: E402

TAU = 1e-3
MULTS = (0.5, 1.0, 1.5, 2.0, 3.0)
BASES = ("F-drift", "F-boot")
BUDGETS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
FDRIFT = ROOT / "runs" / "m2_dev" / "fdrift.json"


# ---------------------------------------------------------------------------------------------------- query instances

def queries(entry: dict, ds, ep) -> dict[str, list[list[int]]]:
    m = len(ds.positions)
    q = {"1": [[i] for i in range(m)]}
    qb = json.loads((ROOT / "qpu" / "M1_qubits.json").read_text(encoding="utf-8"))
    pr = entry["queries"]["2_pairs"]
    if pr.get("rule") == "explicit":
        q["2"] = [list(p) for p in pr["pairs"]]
    else:
        pos_of = {p.qubit: p.index for p in ds.positions}
        pairs = set()
        for a, b in qb["edges_virtual"]:
            qa, qb_ = qb["qubits"][a], qb["qubits"][b]
            if qa in pos_of and qb_ in pos_of:
                pairs.add(tuple(sorted((pos_of[qa], pos_of[qb_]))))
        q["2"] = [list(p) for p in sorted(pairs)]
    q["3"] = [list(s) for k in ("3", "4") for s in entry["queries"]["3_marginals"][k]]
    mr = entry["queries"]["4_mitigated"]
    if mr.get("rule") == "explicit":
        q["4"] = [list(mr["S"])]
    else:
        pos_of = {p.qubit: p.index for p in ds.positions}
        S = [pos_of[qb["qubits"][v]] for v in qb["path_virtual"][:4] if qb["qubits"][v] in pos_of]
        q["4"] = [S] if S else []
    return q


def useful(ls: np.ndarray, S: list[int], cls: str, budget: float, g4: np.ndarray | None = None) -> bool:
    """ls: (N, m) label-set codes. budget = tau * N (single-position bound in counts). g4: the mitigated g = M^T f of a
    class-4 instance under the same vector (required for class 4; exact realized width)."""
    sub = ls[:, S]
    k = len(S)
    if cls == "4":
        from fractions import Fraction
        lo, hi = intervals_multi(sub, [g4], exact=True)[0]
        osc = Fraction(float(np.max(g4))) - Fraction(float(np.min(g4)))  # exact
        return (hi - lo) <= Fraction(k) * Fraction(budget) * osc
    amb_rows = np.flatnonzero((sub == LS_BOTH).any(axis=1))
    if cls == "2" and amb_rows.size > k * budget:
        return False  # parity <Z_S>: width = 2 * #ambiguous rows <= 2 |S| tau N
    if amb_rows.size <= k * budget:
        return True  # every pattern-count width <= #ambiguous rows
    a = sub[amb_rows]
    for pat in range(1 << k):
        ok = np.ones(a.shape[0], dtype=bool)
        for i in range(k):
            ok &= ((a[:, i] >> ((pat >> i) & 1)) & 1).astype(bool)  # LS code bit0: contains 0, bit1: contains 1
        if ok.sum() > k * budget:
            return False
    return True


def class4_g(ds, S: list[int], hs) -> np.ndarray:
    """g = M^T f, f = parity, M = kron of inverse assignment matrices from the ingest-day calibration under hs[j]."""
    A = [assignment_matrix(ds.positions[j].cal.z0, ds.positions[j].cal.z1, ds.positions[j].cal.frame, hs[j]) for j in S]
    return mitigation_g(A, g_parity(len(S)))


# ---------------------------------------------------------------------------------------------------- label sets

def box_region_sets(ref: Discriminator, M0: float, M1: float, box, h: Discriminator) -> list[int]:
    """B4q+box: label sets under h of the two closed regions outside the slab, box and {v <= -M0}, box and {v >= M1},
    v = <u_ref, z> - beta_ref as computed by `b4_slab` in float. Each region is widened by g >= |fl(v) - v| (sound
    superset); its vertices are exact rationals (box corners inside + line / edge intersections); the label set is the
    exact union over vertices (convex closed region: extremes at vertices). Returns [ls0, ls1] (0 = empty region)."""
    from fractions import Fraction as F
    from iqstore.family import unit
    a_, b_ = float(np.cos(ref.theta)), float(np.sin(ref.theta))
    X0, Y0, X1, Y1 = box
    eps = float(np.finfo(np.float64).eps)
    g = 4 * eps * (abs(a_) * max(abs(X0), abs(X1)) + abs(b_) * max(abs(Y0), abs(Y1)) + abs(ref.beta)) + 1e-300
    A, B = F(a_), F(b_)
    corners = [(F(X0), F(Y0)), (F(X1), F(Y0)), (F(X1), F(Y1)), (F(X0), F(Y1))]
    p, q = unit(h.theta)
    P, Q, Bh = F(p), F(q), F(h.beta)
    out = []
    for c, ge in ((F(ref.beta) - F(M0) + F(g), False), (F(ref.beta) + F(M1) - F(g), True)):
        def inside(X, Y, c=c, ge=ge):
            v = A * X + B * Y
            return v >= c if ge else v <= c
        verts = [v for v in corners if inside(*v)]
        for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):
            fa, fb = A * xa + B * ya - c, A * xb + B * yb - c
            if (fa < 0 < fb) or (fb < 0 < fa):
                t_ = fa / (fa - fb)
                verts.append((xa + t_ * (xb - xa), ya + t_ * (yb - ya)))
        if not verts:
            out.append(0)
            continue
        dv = [P * X + Q * Y - Bh for X, Y in verts]
        out.append((LS_ONLY0 if any(d <= 0 for d in dv) else 0) | (LS_ONLY1 if any(d > 0 for d in dv) else 0))
    return out


def b4q_label_sets(x, y, in_slab, side1, info, fam, h: Discriminator, raw_exact: bool, regions=None) -> np.ndarray:
    out = np.empty(x.size, dtype=np.int8)
    in_fam = fam.contains(h)
    if in_fam:
        out[~in_slab] = np.where(side1[~in_slab], LS_ONLY1, LS_ONLY0)
    elif regions is None:
        out[~in_slab] = LS_BOTH
    else:
        out[~in_slab & ~side1] = regions[0] or LS_BOTH
        out[~in_slab & side1] = regions[1] or LS_BOTH
    xs, ys = x[in_slab], y[in_slab]
    if raw_exact:  # B4 fallback: exact slab points
        out[in_slab] = np.where(h.label(xs, ys), LS_ONLY1, LS_ONLY0)
        return out
    d = info["delta"]
    kx = np.floor(xs / d).astype(np.int64)
    ky = np.floor(ys / d).astype(np.int64)
    kx = np.where(xs < kx * d, kx - 1, kx)
    kx = np.where(xs >= (kx + 1) * d, kx + 1, kx)
    ky = np.where(ys < ky * d, ky - 1, ky)
    ky = np.where(ys >= (ky + 1) * d, ky + 1, ky)
    out[in_slab] = rect_label_sets(kx * d, ky * d, (kx + 1) * d, (ky + 1) * d, h.theta, h.beta)
    return out


def b2_label_sets(rx, ry, e: float, h: Discriminator) -> np.ndarray:
    """B2 margin rule: each shot lies in the CLOSED square [r - e, r + e] (r its stored reconstruction, e the measured
    error); exact closed-rectangle label sets."""
    return closed_rect_label_sets(rx - e, ry - e, rx + e, ry + e, h.theta, h.beta)


def b2_at_eps(x, y, frame, codec: str, eps: float):
    """B2 at one eps of the knob grid: (blobs, rx, ry, e), or None if the reconstruction is not finite (reported)."""
    import struct
    from baselines.storage_baselines import _sz3_roundtrip, _zfp_roundtrip
    from iqstore.storage import zbest
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    rnd = float(max(np.max(np.abs(xf.astype(np.float64) - x)), np.max(np.abs(yf.astype(np.float64) - y))))
    rt = _zfp_roundtrip if codec == "zfp" else _sz3_roundtrip
    cx, rx_ = rt(xf, float(eps))
    cy, ry_ = rt(yf, float(eps))
    rx_, ry_ = rx_.astype(np.float64), ry_.astype(np.float64)
    if not (np.all(np.isfinite(rx_)) and np.all(np.isfinite(ry_))):
        return None
    e = float(np.nextafter(float(max(np.max(np.abs(rx_ - xf)), np.max(np.abs(ry_ - yf)))) + rnd, np.inf))
    hdr = struct.pack("<B4ddq", 1 if codec == "zfp" else 2, *frame.to_array(), e, x.size)
    return [hdr, zbest(cx), zbest(cy)], rx_, ry_, e


# ---------------------------------------------------------------------------------------------------- maintenance work

def _iq_cells(rep, fam, h, occupied: np.ndarray):
    """Label sets of IQSTORE's indexed cells under h: (band leaves, their label sets, [decided-0, decided-1] region label
    sets, #cells evaluated). The cell -> shots index lists the band leaves holding at least one shot (`occupied`); an
    empty leaf has nothing to maintain (B4q's stored cells are occupied by construction)."""
    part = rep.part
    band = np.flatnonzero((part.status == -1) & occupied)
    lb = rect_label_sets(part.x0[band], part.y0[band], part.x1[band], part.y1[band], h.theta, h.beta)
    n_eval = band.size + 2
    reg = []
    for s in (0, 1):
        if fam.contains(h):
            reg.append(LS_ONLY1 if s else LS_ONLY0)
        else:
            sel = np.flatnonzero(part.status == s)
            n_eval += sel.size
            ls = rect_label_sets(part.x0[sel], part.y0[sel], part.x1[sel], part.y1[sel], h.theta, h.beta)
            reg.append(int(np.bitwise_or.reduce(ls)) if sel.size else 0)
    return band, lb, reg, n_eval


def maint_iq(rep, fam, versions, n_views: int) -> dict:
    """prereg Q3 maintenance algorithm (identical for IQSTORE and B4q-inc): per version change, evaluate the label set of
    every indexed cell (occupied band leaves and the two decided regions) under old and new h; read the shots of every changed cell
    (cell -> shots index) and update their view entries; re-label exceptions exactly."""
    codes = rep.codes
    band_leaf_shots = np.bincount(codes[codes >= 3] - 3, minlength=rep.part.n_leaves)
    reg_shots = [int((codes == 0).sum()), int((codes == 1).sum())]
    n_exc = int(rep.exc_index.size)
    w = {"cells": 0, "shots": 0, "views": 0, "exceptions": 0}
    occ = band_leaf_shots > 0
    prev = _iq_cells(rep, fam, versions[0], occ)
    for h in versions[1:]:
        cur = _iq_cells(rep, fam, h, occ)
        w["cells"] += prev[3] + cur[3]
        ch = cur[1] != prev[1]
        touched = int(band_leaf_shots[cur[0][ch]].sum()) + sum(r for r, a, b in zip(reg_shots, prev[2], cur[2]) if a != b)
        w["shots"] += touched
        w["exceptions"] += n_exc
        w["views"] += (touched + n_exc) * n_views
        prev = cur
    return w


def _b4q_cells(x, y, in_slab, qinfo, raw_exact: bool):
    xs, ys = x[in_slab], y[in_slab]
    if raw_exact:
        return xs, ys, xs, ys, np.ones(xs.size, dtype=np.int64)  # exact points (degenerate cells)
    d = qinfo["delta"]
    kx = np.floor(xs / d).astype(np.int64)
    ky = np.floor(ys / d).astype(np.int64)
    kx = np.where(xs < kx * d, kx - 1, kx)
    kx = np.where(xs >= (kx + 1) * d, kx + 1, kx)
    ky = np.where(ys < ky * d, ky - 1, ky)
    ky = np.where(ys >= (ky + 1) * d, ky + 1, ky)
    uk, cnt = np.unique(np.stack([kx, ky], 1), axis=0, return_counts=True)
    return uk[:, 0] * d, uk[:, 1] * d, (uk[:, 0] + 1) * d, (uk[:, 1] + 1) * d, cnt


def maint_b4q(x, y, in_slab, side1, qinfo, fam, versions, n_views: int, box_regions=None) -> dict:
    exact = qinfo["mode"] != "grid"
    x0, y0, x1, y1, cnt = _b4q_cells(x, y, in_slab, qinfo, exact)
    reg_shots = [int((~in_slab & ~side1).sum()), int((~in_slab & side1).sum())]
    w = {"cells": 0, "shots": 0, "views": 0, "exceptions": 0}

    def cells(h):
        ls = (np.where(h.label(x0, y0), LS_ONLY1, LS_ONLY0) if exact else
              rect_label_sets(x0, y0, x1, y1, h.theta, h.beta))
        reg = ([LS_ONLY0, LS_ONLY1] if fam.contains(h) else
               [LS_BOTH, LS_BOTH] if box_regions is None else box_regions(h))
        return ls, reg, x0.size + 2

    prev = cells(versions[0])
    for h in versions[1:]:
        cur = cells(h)
        w["cells"] += prev[2] + cur[2]
        ch = cur[0] != prev[0]
        touched = int(cnt[ch].sum()) + sum(r for r, a, b in zip(reg_shots, prev[1], cur[1]) if a != b)
        w["shots"] += touched
        w["views"] += touched * n_views
        prev = cur
    return w


# ---------------------------------------------------------------------------------------------------- run

def later_refs(eps: dict, ds) -> dict[int, list[Discriminator]]:
    out = {}
    for e2 in M.DEV_EPOCHS:
        if e2 <= ds.epoch:
            continue
        hs = []
        for p in ds.positions:
            c2 = eps[e2].cal[(p.qubit, p.context)]
            x0, y0 = p.cal.frame.apply(c2.z0)
            x1, y1 = p.cal.frame.apply(c2.z1)
            hs.append(fit_lda(x0, y0, x1, y1))
        out[e2] = hs
    return out


def later_refs_all(ds) -> dict[int, list[Discriminator]]:
    """TEST use (after the freeze): later-day vectors (b) from every ACCEPTED later campaign epoch."""
    out = {}
    for e2 in sorted(M.ledger_jobs()):
        if e2 <= ds.epoch:
            continue
        ep2 = M.load_epoch(e2, allow_test=True)
        hs = []
        for p in ds.positions:
            c2 = ep2.cal[(p.qubit, p.context)]
            x0, y0 = p.cal.frame.apply(c2.z0)
            x1, y1 = p.cal.frame.apply(c2.z1)
            hs.append(fit_lda(x0, y0, x1, y1))
        out[e2] = hs
    return out


def base_half(p, base: str, fd: dict) -> tuple[float, float]:
    if base == "F-drift":
        c = fd["contexts"][p.context]
        return c["dtheta_half_width"], c["dbeta_half_width"]
    f = p.fams["F-boot"]
    return 0.5 * (f.th_hi - f.th_lo), 0.5 * (f.b_hi - f.b_lo)


def _code_commit() -> str:
    from eval.m2_dev import provenance
    return json.dumps(provenance(), sort_keys=True)  # full provenance (commit, environment, DEV jobs)


def _load_all(out: pathlib.Path) -> dict:
    """Merge cov.json and every shard file cov_s<k>of<n>.json; a dataset present twice must be identical."""
    D = {}
    for f in sorted(out.glob("cov*.json")):
        if f.name == "cov_summary.json":
            continue
        for n, row in json.loads(f.read_text(encoding="utf-8"))["datasets"].items():
            if n in D and D[n] != row:
                raise RuntimeError(f"dataset {n} appears twice with different content")
            D[n] = row
    return D


WIDTH_METHODS = ("IQSTORE", "B4q", "B4q+box", "A7")
B2_METHODS = ("B2-zfp", "B2-SZ3")


def run(out: pathlib.Path, shard: tuple[int, int] = (0, 1)) -> None:
    iq = json.loads((out / "iqgrid.json").read_text(encoding="utf-8"))["selected"]
    fd = json.loads(FDRIFT.read_text(encoding="utf-8"))
    k_sh, n_sh = shard
    path = out / ("cov.json" if n_sh == 1 else f"cov_s{k_sh}of{n_sh}.json")
    loaded = _load_all(out)
    done = set(loaded)
    commit = _code_commit()
    if "+dirty" in commit:
        raise RuntimeError("commit the code first (provenance)")
    for n, r_ in loaded.items():  # every shard's rows, not only this shard's
        if r_["code_commit"] != commit:
            raise RuntimeError(f"{n} in {out} has different provenance; use a fresh --out")
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": {}}
    for n, r_ in res["datasets"].items():
        if r_["code_commit"] != commit:
            raise RuntimeError(f"{path.name}: {n} has different provenance; use a fresh --out")
    store = out / "containers"
    store.mkdir(parents=True, exist_ok=True)
    w = {e["id"]: e for e in M.workload()["datasets"]}
    eps = {e: M.load_epoch(e) for e in M.DEV_EPOCHS}
    for i_ds, (ep, ds) in enumerate(M.dev_datasets()):
        if ds.name in res["datasets"] or ds.name in done or i_ds % n_sh != k_sh:
            continue
        t0 = time.time()
        N, m = ds.positions[0].raw.size, len(ds.positions)
        budget = TAU * N
        qs = queries(w[ds.name], ds, ep)
        refs = later_refs(eps, ds)
        g4 = {e2: [class4_g(ds, S, hs) for S in qs["4"]] for e2, hs in refs.items()}
        xy = [p.cal.frame.apply(p.raw) for p in ds.positions]
        nv = [sum(j in S for Sx in qs.values() for S in Sx) for j in range(m)]
        row = {"code_commit": commit, "iqstore_settings": iq, "tau": TAU, "epoch": ds.epoch, "N": N, "m": m,
               "n_queries": {c: len(v) for c, v in qs.items()}, "later_epochs": sorted(refs), "configs": {}}

        def use_counts(lsm):
            return {str(e2): {c: int(sum(useful(lsm[e2], S, c, budget, g4[e2][ii] if c == "4" else None)
                                         for ii, S in enumerate(Sx)))
                              for c, Sx in qs.items()} for e2 in refs}

        # ---- width-knob methods
        for base in BASES:
            for mult in MULTS:
                t1 = time.time()
                blobs = {"IQSTORE": [], "B4q": [], "A7": []}
                ls = {meth: {e2: np.empty((N, m), dtype=np.int8) for e2 in refs} for meth in WIDTH_METHODS}
                infam = {e2: 0 for e2 in refs}
                pos_u = {meth: {e2: [] for e2 in refs} for meth in WIDTH_METHODS}
                st = {"iq_band": 0, "b4q_slab": 0, "b4q_fallback": 0, "iq_factor": [], "a7_fallback": 0, "a7_k": []}
                wm = {meth: {"cells": 0, "shots": 0, "views": 0, "exceptions": 0} for meth in WIDTH_METHODS}
                for j, (p, (x, y)) in enumerate(zip(ds.positions, xy)):
                    hw = base_half(p, base, fd)
                    fam = family_from_widths(p.cal.ref, mult * hw[0], mult * hw[1])
                    rep, ib, info = ingest_position(p.cal.frame, fam, x, y, p.raw, TAU, iq["d0"], iq["max_depth"],
                                                    iq["split_fraction"])
                    blobs["IQSTORE"] += _grouped(ib)
                    st["iq_band"] += int((rep.codes >= 3).sum())
                    st["iq_factor"].append(info["factor"])
                    box = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
                    M0, M1 = fixed_margins(p.cal.ref, fam, box)
                    in_slab, side1 = b4_slab(x, y, p.cal.ref, M0, M1)
                    breg = (lambda h, M0=M0, M1=M1, box=box, ref=p.cal.ref: box_region_sets(ref, M0, M1, box, h))
                    qb, qinfo = b4q_position(x, y, in_slab, side1, p.raw, p.cal.frame, p.cal.ref, M0, M1, fam, TAU,
                                             DELTAS)
                    blobs["B4q"] += _grouped(qb)
                    a7rep, a7b, a7i = a7_position(p.cal.frame, fam, x, y, p.raw, TAU)
                    blobs["A7"] += _grouped(a7b)
                    st["a7_fallback"] += int(a7rep is None)
                    st["a7_k"].append(a7i.get("k"))
                    st["b4q_slab"] += int(in_slab.sum())
                    st["b4q_fallback"] += int(qinfo["mode"] != "grid")
                    if refs:  # real event stream inside DEV: reference -> refit of each later DEV epoch, in order
                        versions = [p.cal.ref] + [refs[e2][j] for e2 in sorted(refs)]
                        for meth, wv in (("IQSTORE", maint_iq(rep, fam, versions, nv[j])),
                                         ("B4q", maint_b4q(x, y, in_slab, side1, qinfo, fam, versions, nv[j])),
                                         ("B4q+box", maint_b4q(x, y, in_slab, side1, qinfo, fam, versions, nv[j], breg)),
                                         ("A7", maint_iq(a7rep, fam, versions, nv[j]) if a7rep is not None else
                                          {"cells": 0, "shots": N * (len(versions) - 1), "exceptions": 0,
                                           "views": N * (len(versions) - 1) * nv[j]})):
                            for k2, v2 in wv.items():
                                wm[meth][k2] += v2
                    for e2, hs in refs.items():
                        h = hs[j]
                        infam[e2] += int(fam.contains(h))
                        a = rep.label_set_codes(h)
                        b = b4q_label_sets(x, y, in_slab, side1, qinfo, fam, h, qinfo["mode"] != "grid")
                        c = b if fam.contains(h) else b4q_label_sets(x, y, in_slab, side1, qinfo, fam, h,
                                                                       qinfo["mode"] != "grid", breg(h))
                        a7 = (a7rep.label_set_codes(h) if a7rep is not None else
                              np.where(h.label(x, y), LS_ONLY1, LS_ONLY0).astype(np.int8))
                        for meth, v in (("IQSTORE", a), ("B4q", b), ("B4q+box", c), ("A7", a7)):
                            ls[meth][e2][:, j] = v
                            pos_u[meth][e2].append(int((v == LS_BOTH).sum()))
                bits = {meth: _measure(store, f"{meth}_{ds.name}_s{k_sh}_cov.bin", bl, N, m, False)
                        for meth, bl in blobs.items()}
                bits["B4q+box"] = bits["B4q"] + 8 * 32 * m / (N * m)  # + 4 float64 per position (the data box)
                row["configs"][f"{base}|{mult:g}"] = {
                    "bits": bits, "useful": {meth: use_counts(ls[meth]) for meth in WIDTH_METHODS},
                    "in_family_positions": {str(e2): v for e2, v in infam.items()},
                    "pos_U": {meth: {str(e2): v for e2, v in d.items()} for meth, d in pos_u.items()},
                    "maintenance": {meth: d | {"w": d["cells"] + d["shots"] + d["exceptions"]} for meth, d in wm.items()},
                    "stats": st, "seconds": round(time.time() - t1, 1)}
                print(ds.name, base, mult, "IQ %.4f B4q %.4f A7 %.4f" % (bits["IQSTORE"], bits["B4q"], bits["A7"]),
                      round(time.time() - t1, 1), "s", flush=True)
        # ---- B2 eps knob (family-independent)
        t1 = time.time()
        for codec in ("zfp", "SZ3"):
            for ie, eps_v in enumerate(EPS_GRID):
                key = f"B2-{codec}|{ie}"
                bl = []
                lsm = {e2: np.empty((N, m), dtype=np.int8) for e2 in refs}
                n_fb = 0
                for j, (p, (x, y)) in enumerate(zip(ds.positions, xy)):
                    r_ = b2_at_eps(x, y, p.cal.frame, codec.lower(), float(eps_v))
                    if r_ is None:  # non-finite reconstruction: this position falls back to exact storage (counted)
                        import struct
                        from iqstore.storage import DTYPE_CODE, zbest
                        raw = np.ascontiguousarray(p.raw)
                        bl += _grouped([struct.pack("<B4dqB", 0, *p.cal.frame.to_array(), N, DTYPE_CODE[raw.dtype]),
                                        zbest(raw.view(np.float64))])
                        for e2, hs in refs.items():
                            lsm[e2][:, j] = np.where(hs[j].label(x, y), LS_ONLY1, LS_ONLY0)
                        n_fb += 1
                        continue
                    blobs_j, rx_, ry_, e = r_
                    bl += _grouped(blobs_j)
                    for e2, hs in refs.items():
                        lsm[e2][:, j] = b2_label_sets(rx_, ry_, e, hs[j])
                bits = _measure(store, f"B2_{ds.name}_s{k_sh}_cov.bin", bl, N, m, False)
                row["configs"][key] = {"bits": {f"B2-{codec}": bits}, "eps": float(eps_v), "exact_fallbacks": n_fb,
                                       "useful": {f"B2-{codec}": use_counts(lsm)}}
        print(ds.name, "B2 eps sweep", round(time.time() - t1, 1), "s", flush=True)
        row["seconds_total"] = round(time.time() - t0, 1)
        res["datasets"][ds.name] = row
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- summary

def _cov(cfg: dict, meth: str, n_q: dict) -> tuple[dict, dict]:
    """Per-class (useful, total) pooled over the later epochs of one dataset."""
    u, t = {}, {}
    for e2, byc in cfg["useful"][meth].items():
        for c, v in byc.items():
            u[c] = u.get(c, 0) + v
            t[c] = t.get(c, 0) + n_q[c]
    return u, t


def _pooled(rows: list[tuple[dict, dict]]) -> float:
    """Useful-query coverage: per class pooled share, classes weighted equally (prereg §5)."""
    u, t = {}, {}
    for uu, tt in rows:
        for c in tt:
            u[c] = u.get(c, 0) + uu.get(c, 0)
            t[c] = t.get(c, 0) + tt[c]
    cls = [c for c in t if t[c] > 0]
    return float(np.mean([u[c] / t[c] for c in cls])) if cls else float("nan")


METHODS = WIDTH_METHODS + B2_METHODS
COMPETITORS = ("B4q", "B4q+box", "A7", "B2-zfp", "B2-SZ3")


def _keys(meth: str, base: str) -> list[str]:
    if meth in B2_METHODS:
        return [f"{meth}|{i}" for i in range(len(EPS_GRID))]
    return [f"{base}|{m:g}" for m in MULTS]


def _bits(D, n, k, meth):
    c = D[n]["configs"][k]
    return c["bits"][meth] if "bits" in c else None


def _matched(D, names, base: str, kq: str, comp: str) -> dict:
    """Comparison 2 (competitor oracle), paired per dataset: IQSTORE at kq vs the competitor's best-coverage grid
    configuration with actual b-bar <= 1.02 x IQSTORE's actual b-bar on that dataset; statistic = median over matched
    datasets of the per-dataset coverage difference; budget inconclusive if more than half the pairs are unmatched."""
    deltas, unmatched = [], []
    for n in names:
        b_iq = _bits(D, n, kq, "IQSTORE")
        cand = [k for k in _keys(comp, base) if _bits(D, n, k, comp) is not None and _bits(D, n, k, comp) <= 1.02 * b_iq]
        if not cand:
            vals = [v for v in (_bits(D, n, k, comp) for k in _keys(comp, base)) if v is not None]
            unmatched.append({"dataset": n, "b_IQ": b_iq, "b_comp_min": min(vals) if vals else None})
            continue
        kb = max(cand, key=lambda k: (_pooled([_cov(D[n]["configs"][k], comp, D[n]["n_queries"])]),
                                      -_bits(D, n, k, comp)))
        ci = _pooled([_cov(D[n]["configs"][kq], "IQSTORE", D[n]["n_queries"])])
        cb = _pooled([_cov(D[n]["configs"][kb], comp, D[n]["n_queries"])])
        deltas.append({"dataset": n, "comp_config": kb, "cov_IQ": ci, "cov_comp": cb, "delta": ci - cb})
    inconclusive = len(unmatched) > len(names) / 2
    return {"iq_config": kq, "matched": len(deltas), "unmatched": unmatched, "inconclusive": inconclusive,
            "median_delta_c": float(np.median([d["delta"] for d in deltas])) if deltas else None, "pairs": deltas}


def summary(out: pathlib.Path) -> dict:
    D = _load_all(out)
    dev = [e["id"] for e in M.workload()["datasets"] if e["split"] == "DEV"]
    missing = [n for n in dev if n not in D]
    if missing:
        raise RuntimeError(f"summary refused: DEV datasets missing {missing}")
    names = dev
    commits = {D[n]["code_commit"] for n in names}
    settings = {json.dumps(D[n]["iqstore_settings"], sort_keys=True) for n in names}
    if len(commits) != 1 or any("+dirty" in c for c in commits) or len(settings) != 1:
        raise RuntimeError(f"summary refused: mixed or dirty provenance {commits} {settings}")
    for n in names:
        for base in BASES:
            for meth in METHODS:
                for k in _keys(meth, base):
                    if k not in D[n]["configs"]:
                        raise RuntimeError(f"summary refused: {n} lacks {k}")
    with_later = [n for n in names if D[n]["later_epochs"]]
    S = {"datasets": len(names), "datasets_with_later_vectors": len(with_later), "code_commit": commits.pop(),
         "iqstore_settings": json.loads(settings.pop()), "provenance_note": "code_commit is the full provenance JSON",
         "maintenance_note": "w values are DEV ESTIMATES from cell populations (not an executed index traversal); "
                             "not used for W3 (c2)",
         "note": "DEV only; coverage uses later DEV epochs; B4q+box is a sensitivity variant", "bases": {}}
    for base in BASES:
        tab = {}
        for meth in METHODS:
            for k in _keys(meth, base):
                vals = [_bits(D, n, k, meth) for n in names]
                if any(v is None for v in vals):
                    continue
                tab.setdefault(k, {})[meth] = {
                    "median_bits": float(np.median(vals)),
                    "useful_coverage": _pooled([_cov(D[n]["configs"][k], meth, D[n]["n_queries"]) for n in with_later])}
        for k in _keys("IQSTORE", base):
            row = tab[k]
            row["in_family_share"] = float(sum(sum(D[n]["configs"][k]["in_family_positions"].values())
                                               for n in with_later) /
                                           sum(D[n]["m"] * len(D[n]["later_epochs"]) for n in with_later))
            for comp in ("B4q", "B4q+box", "A7"):
                ra = [D[n]["configs"][k]["bits"][comp] / D[n]["configs"][k]["bits"]["IQSTORE"] for n in names]
                rc2 = [D[n]["configs"][k]["maintenance"][comp]["w"] / D[n]["configs"][k]["maintenance"]["IQSTORE"]["w"]
                       for n in with_later if D[n]["configs"][k]["maintenance"]["IQSTORE"]["w"] > 0]
                row[f"r_a_median|{comp}"] = float(np.median(ra))
                row[f"r_c2_median|{comp}"] = float(np.median(rc2)) if rc2 else None
                row[f"r_c2_per_dataset|{comp}"] = rc2
            for comp in COMPETITORS:
                row[f"matched_at_this_IQ_config|{comp}"] = _matched(D, with_later, base, k, comp)
        dep = {}
        for meth in METHODS:
            dep[meth] = {}
            keys = [k for k in _keys(meth, base) if k in tab and meth in tab[k]]
            for bstar in BUDGETS:
                fit = [k for k in keys if tab[k][meth]["median_bits"] <= bstar]
                # ties: smaller DEV median b-bar, then smaller width (smaller eps index = finer for B2: listed first)
                dep[meth][str(bstar)] = (max(fit, key=lambda k: (tab[k][meth]["useful_coverage"], -tab[k][meth]["median_bits"],
                                                                 -keys.index(k))) if fit else None)
        mat = {comp: {str(bs): (_matched(D, with_later, base, dep["IQSTORE"][str(bs)], comp)
                                if dep["IQSTORE"][str(bs)] else None) for bs in BUDGETS} for comp in COMPETITORS}
        S["bases"][base] = {"grid": tab, "deployed": dep, "matched_deployed": mat}
    (out / "cov_summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    return S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["run", "summary"])
    ap.add_argument("--out", default=str(ROOT / "runs" / "m2_dev" / "r2"))
    ap.add_argument("--shard", default="0/1", help="k/n: datasets with workload-DEV order index i, i mod n == k")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "run":
        run(out, tuple(int(v) for v in a.shard.split("/")))
    else:
        print(json.dumps(summary(out), indent=1))


if __name__ == "__main__":
    main()
