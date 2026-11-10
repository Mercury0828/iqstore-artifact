"""A6: Hazy-style view maintenance on raw IQ points (prereg/M2.md v4 section 3, "A6"; Koc & Re, PVLDB 4(5) 2011, adapted).

Per position: the raw points (frame coordinates), an anchor discriminator h_a with float64 coefficients (p_a, q_a), the
anchor distances d~_s = fl(fl(fl(p_a x) + fl(q_a y)) - beta_a) sorted with their permutation, and per block of B sorted
shots outward-rounded bounds R_blk >= max |z_s| and E_blk >= the float-filter error of d~_s (identical formulas to
`baselines.storage_baselines.a6_blobs`, which writes the same arrays to disk).

Version change h_old -> h_new at position j (eager):
  bound   Delta(h, r) = |beta - beta_a| + r * ||(p - p_a, q - q_a)||_2; the two scalars |beta - beta_a| and the norm are
          evaluated in exact rationals and rounded up; products and sums with r are rounded outward;
  step 1  binary search on the sorted d~ for |d~| <= D(R) + E, D(r) = max(Delta(h_old, r), Delta(h_new, r));
  step 2  inside that range, the blocks with min |d~| <= D(R_blk) + E_blk;
  step 3  every candidate re-labelled exactly under h_old and h_new (float filter, exact rational fallback);
  views   every view touching j: old pattern subtracted and new pattern added for the shots whose label changed.
Re-anchoring (Hazy's rule): when the candidates inspected at a position since its last anchor reach N, the anchor becomes
the current discriminator and d~, the permutation and the block bounds are recomputed; charged to the event.
Views hold exact pattern counts and the parity sum (A6 keeps raw points, so its answers are exact).
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

import numba
import numpy as np

from iqstore.family import Discriminator, exact_sign, unit

_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)
KMAX = 4


def _up_frac(v: Fraction) -> float:
    f = float(v)
    while Fraction(f) < v:
        f = math.nextafter(f, math.inf)
    return f


def _delta_terms(p: float, q: float, beta: float, pa: float, qa: float, ba: float) -> tuple[float, float]:
    """(upper bound of |beta - beta_a|, upper bound of ||(p - p_a, q - q_a)||_2), both from exact rationals."""
    db = _up_frac(abs(Fraction(beta) - Fraction(ba)))
    s2 = (Fraction(p) - Fraction(pa)) ** 2 + (Fraction(q) - Fraction(qa)) ** 2
    r = math.nextafter(math.sqrt(_up_frac(s2)), math.inf)
    while Fraction(r) ** 2 < s2:
        r = math.nextafter(r, math.inf)
    return db, r


def anchor_arrays(x: np.ndarray, y: np.ndarray, p: float, q: float, beta: float, block: int):
    """Sorted d~, permutation, per-block R and E (the formulas of `a6_blobs`)."""
    d = (p * x + q * y) - beta
    perm = np.argsort(d, kind="stable").astype(np.int64)
    ds_ = d[perm]
    up = lambda v: np.nextafter(v, np.inf)  # noqa: E731
    s2 = up(up(up(x * x) + up(y * y)) * (1 + 4 * _EPS))
    r = up(up(np.sqrt(s2)) * (1 + 4 * _EPS))[perm]
    e = up(up(2 * _EPS * up(up(up(np.abs(p * x)) + up(np.abs(q * y))) + abs(beta)) * (1 + 4 * _EPS)) + _TINY)[perm]
    nb = -(-x.size // block)
    pad = nb * block - x.size
    if pad:
        r = np.concatenate([r, np.full(pad, -np.inf)])
        e = np.concatenate([e, np.full(pad, -np.inf)])
    R = np.nextafter(r.reshape(nb, block).max(axis=1), np.inf)
    E = np.nextafter(e.reshape(nb, block).max(axis=1), np.inf)
    return ds_, perm, R, E


@numba.njit(cache=True)
def _uprod(a, b):
    return np.nextafter(a * b, np.inf)


@numba.njit(cache=True)
def _usum(a, b):
    return np.nextafter(a + b, np.inf)


@numba.njit(cache=True)
def _candidates(ds_, perm, R, E, block, db_o, nr_o, db_n, nr_n, out):
    """Steps 1-2. Returns (#candidates written to `out`, #blocks evaluated, #binary-search probes)."""
    N = ds_.size
    Rg = R.max()
    Eg = E.max()
    T = _usum(max(_usum(db_o, _uprod(Rg, nr_o)), _usum(db_n, _uprod(Rg, nr_n))), Eg)
    # step 1: binary search for the range |d~| <= T
    probes = 0
    lo, hi = 0, N
    while lo < hi:
        mid = (lo + hi) >> 1
        probes += 1
        if ds_[mid] < -T:
            lo = mid + 1
        else:
            hi = mid
    a = lo
    lo, hi = a, N
    while lo < hi:
        mid = (lo + hi) >> 1
        probes += 1
        if ds_[mid] <= T:
            lo = mid + 1
        else:
            hi = mid
    b = lo
    nc = 0
    nblk = 0
    if a >= b:
        return 0, 0, probes
    # step 2: block filter inside [a, b)
    for k in range(a // block, (b - 1) // block + 1):
        s0 = max(k * block, a)
        s1 = min((k + 1) * block, b)
        nblk += 1
        f, l = ds_[s0], ds_[s1 - 1]
        mn = 0.0 if (f <= 0.0 <= l) else min(abs(f), abs(l))
        thr = _usum(max(_usum(db_o, _uprod(R[k], nr_o)), _usum(db_n, _uprod(R[k], nr_n))), E[k])
        if mn <= thr:
            for i in range(s0, s1):
                out[nc] = perm[i]
                nc += 1
    return nc, nblk, probes


@numba.njit(cache=True)
def _sign(x, y, p, q, b):
    px = p * x
    qy = q * y
    d = (px + qy) - b
    err = 2.0 * _EPS * (abs(px) + abs(qy) + abs(b)) + _TINY
    if not (abs(d) > err):
        return 2
    return 1 if d > 0 else 0


@numba.njit(cache=True)
def _relabel(x, y, cand, nc, po, qo, bo, pn, qn, bn, lab_old, lab_new, unsure):
    """Step 3 (float filter): labels under h_old and h_new; 2 = unsettled (exact fallback in Python)."""
    nu = 0
    for i in range(nc):
        s = cand[i]
        lab_old[i] = _sign(x[s], y[s], po, qo, bo)
        lab_new[i] = _sign(x[s], y[s], pn, qn, bn)
        if lab_old[i] == 2 or lab_new[i] == 2:
            unsure[nu] = i
            nu += 1
    return nu


@numba.njit(cache=True)
def _apply_views(L, F, fptr, fidx, J, vS, vk, vptr, vidx, cnt, par, vmark, mark, buf):
    """Views touching J: for the union of their flipped shots, subtract the old pattern and add the new one; then L ^= F
    on the flipped entries and F is cleared. Returns view-shot updates."""
    m = L.shape[0]
    upd = 0
    slot = np.full(m, -1, np.int64)
    for a in range(J.size):
        slot[J[a]] = a
    vl = np.empty(max(vidx.size, 1), np.int64)
    nv = 0
    for a in range(J.size):
        for q in range(vptr[J[a]], vptr[J[a] + 1]):
            v = vidx[q]
            if not vmark[v]:
                vmark[v] = True
                vl[nv] = v
                nv += 1
    for t in range(nv):
        v = vl[t]
        vmark[v] = False
        k = vk[v]
        nb = 0
        for i in range(k):
            a = slot[vS[v, i]]
            if a < 0:
                continue
            for q in range(fptr[a], fptr[a + 1]):
                s = fidx[q]
                if not mark[s]:
                    mark[s] = True
                    buf[nb] = s
                    nb += 1
        for q in range(nb):
            s = buf[q]
            mark[s] = False
            po = 0
            pn = 0
            for i in range(k):
                j = vS[v, i]
                lo = L[j, s]
                po |= lo << i
                pn |= (lo ^ F[j, s]) << i
            cnt[v, po] -= 1
            cnt[v, pn] += 1
            if po != pn:
                so = 1
                sn = 1
                for i in range(k):
                    if (po >> i) & 1:
                        so = -so
                    if (pn >> i) & 1:
                        sn = -sn
                par[v] += sn - so
            upd += 1
    for a in range(J.size):
        j = J[a]
        for q in range(fptr[a], fptr[a + 1]):
            s = fidx[q]
            L[j, s] ^= F[j, s]
            F[j, s] = 0
    return upd


@numba.njit(cache=True)
def recount(L, vS, vk, cnt, par):
    """Exact view counts from the label matrix (check only; untimed)."""
    V = vS.shape[0]
    N = L.shape[1]
    cnt[:] = 0
    par[:] = 0
    for v in range(V):
        k = vk[v]
        for s in range(N):
            pt = 0
            for i in range(k):
                pt |= L[vS[v, i], s] << i
            cnt[v, pt] += 1
            sg = 1
            for i in range(k):
                if (pt >> i) & 1:
                    sg = -sg
            par[v] += sg


class A6:
    """A6 state of one dataset: raw points, anchors, labels under the current version, exact views."""

    def __init__(self, xs: list[np.ndarray], ys: list[np.ndarray], h0: list[Discriminator], view_sets, block: int,
                 vptr: np.ndarray, vidx: np.ndarray):
        self.m, self.N = len(xs), xs[0].size
        self.x, self.y = xs, ys
        self.block = block
        self.h = list(h0)
        self.anchor = [None] * self.m
        for j in range(self.m):
            self._reanchor(j, h0[j])
        self.L = np.ascontiguousarray(np.stack([h0[j].label(xs[j], ys[j]) for j in range(self.m)]).astype(np.uint8))
        self.F = np.zeros_like(self.L)
        V = len(view_sets)
        self.vS = np.full((V, KMAX), -1, np.int64)
        self.vk = np.zeros(V, np.int64)
        for v, s in enumerate(view_sets):
            self.vS[v, :len(s)] = s
            self.vk[v] = len(s)
        self.vptr, self.vidx = vptr, vidx
        self.cnt = np.zeros((V, 1 << KMAX), np.int64)
        self.par = np.zeros(V, np.int64)
        recount(self.L, self.vS, self.vk, self.cnt, self.par)
        self.vmark = np.zeros(V, np.bool_)
        self.mark = np.zeros(self.N, np.bool_)
        self.buf = np.empty(self.N, np.int64)
        self.cand = np.empty(self.N, np.int64)

    def _reanchor(self, j: int, h: Discriminator) -> None:
        p, q = unit(h.theta)
        ds_, perm, R, E = anchor_arrays(self.x[j], self.y[j], p, q, h.beta, self.block)
        self.anchor[j] = {"p": p, "q": q, "b": h.beta, "ds": ds_, "perm": perm, "R": R, "E": E, "inspected": 0}

    def snapshot(self):
        return (self.L.copy(), self.cnt.copy(), self.par.copy(), list(self.h),
                [dict(a) for a in self.anchor])

    def restore(self, snap) -> None:
        L, c, p, h, an = snap
        self.L[:] = L
        self.cnt[:] = c
        self.par[:] = p
        self.h = list(h)
        self.anchor = [dict(a) for a in an]

    def event(self, J: list[int], hs_new: list[Discriminator]) -> tuple[float, dict]:
        """Eager maintenance of one version change at J. Returns (seconds, work counts)."""
        t0 = time.perf_counter()
        w = {"candidates": 0, "blocks": 0, "probes": 0, "flips": 0, "reanchors": 0, "exact_fallbacks": 0}
        fl_lists = []
        for j in J:
            an = self.anchor[j]
            ho, hn = self.h[j], hs_new[j]
            po, qo = unit(ho.theta)
            pn, qn = unit(hn.theta)
            db_o, nr_o = _delta_terms(po, qo, ho.beta, an["p"], an["q"], an["b"])
            db_n, nr_n = _delta_terms(pn, qn, hn.beta, an["p"], an["q"], an["b"])
            nc, nblk, probes = _candidates(an["ds"], an["perm"], an["R"], an["E"], self.block, db_o, nr_o, db_n, nr_n,
                                           self.cand)
            cand = self.cand[:nc]
            lo = np.empty(nc, np.int8)
            ln = np.empty(nc, np.int8)
            uns = np.empty(nc, np.int64)
            nu = _relabel(self.x[j], self.y[j], cand, nc, po, qo, ho.beta, pn, qn, hn.beta, lo, ln, uns)
            if nu:
                u = uns[:nu]
                s = cand[u]
                lo[u] = exact_sign(self.x[j][s], self.y[j][s], po, qo, ho.beta) > 0
                ln[u] = exact_sign(self.x[j][s], self.y[j][s], pn, qn, hn.beta) > 0
                w["exact_fallbacks"] += int(nu)
            ch = cand[lo != ln]
            self.F[j, ch] = 1
            fl_lists.append(ch)
            w["candidates"] += int(nc)
            w["blocks"] += int(nblk)
            w["probes"] += int(probes)
            w["flips"] += int(ch.size)
            an["inspected"] += int(nc)
            if an["inspected"] >= self.N:
                self._reanchor(j, hn)
                w["reanchors"] += 1
        Ja = np.array(J, dtype=np.int64)
        fptr = np.concatenate([[0], np.cumsum([a.size for a in fl_lists])]).astype(np.int64)
        fidx = np.concatenate(fl_lists).astype(np.int64) if fptr[-1] else np.zeros(1, np.int64)
        w["view_shot_updates"] = int(_apply_views(self.L, self.F, fptr, fidx, Ja, self.vS, self.vk, self.vptr, self.vidx,
                                                  self.cnt, self.par, self.vmark, self.mark, self.buf))
        for j in J:
            self.h[j] = hs_new[j]
        return time.perf_counter() - t0, w

    def check(self, views_iq=None) -> None:
        """Untimed: exact recount from labels recomputed from the raw points under the current version; and, if given,
        containment of the exact counts in IQSTORE's intervals (lo <= count <= hi; parity inside its interval)."""
        L = np.ascontiguousarray(np.stack([self.h[j].label(self.x[j], self.y[j]) for j in range(self.m)])
                                 .astype(np.uint8))
        if not np.array_equal(L, self.L):
            raise AssertionError("A6 labels differ from an exact relabel")
        c = np.zeros_like(self.cnt)
        p = np.zeros_like(self.par)
        recount(L, self.vS, self.vk, c, p)
        if not (np.array_equal(c, self.cnt) and np.array_equal(p, self.par)):
            raise AssertionError("A6 views differ from an exact recount")
        if views_iq is not None:
            P = 1 << self.vk
            for v in range(self.vS.shape[0]):
                lo, hi = views_iq.lo[v, :P[v]], views_iq.hi[v, :P[v]]
                if np.any(self.cnt[v, :P[v]] < lo) or np.any(self.cnt[v, :P[v]] > hi):
                    raise AssertionError(f"exact count outside the IQSTORE interval (view {v})")
                if not (views_iq.par[v, 0] <= self.par[v] <= views_iq.par[v, 1]):
                    raise AssertionError(f"exact parity outside the IQSTORE interval (view {v})")
