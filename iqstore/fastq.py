"""Warm IQSTORE query path for pattern-count statistics (classes 1-3, class 5).

Per position, decoded once into memory:
  side  packed uint64 words, bit s = 1 iff shot s is decided with side 1;
  dec   packed uint64 words, bit s = 1 iff shot s is decided (not band, not exception);
  the band shots (indices, leaf ids) and the exceptions (indices, exact frame coordinates).
A query (S, h_S) returns, for every label pattern p over S, the exact interval [lo_p, hi_p] of the pattern count, and the
parity <Z_S> interval (integers). Shots decided at every position of S are counted by popcount over AND-ed words (one
numba pass for all 2^k patterns): their label tuple is a single pattern if each decided region's label set under h_j is a
singleton (always true for h_j in the family; otherwise the region label set is computed from its leaves). The other
shots (band or exception at some position of S) are few; their label-set tuples are evaluated exactly per shot.
The result equals `intervals_multi` on the full label-set matrix (tested), with integer arithmetic.
"""
from __future__ import annotations

import itertools

import numba
import numpy as np

from .family import Discriminator
from .rep import CODE_EXCEPTION, CODE_OMEGA0, CODE_OMEGA1, LS_BOTH, LS_ONLY0, LS_ONLY1, PositionRep, rect_label_sets


def _pack64(mask: np.ndarray) -> np.ndarray:
    n = mask.size
    w = -(-n // 64)
    b = np.zeros(w * 64, dtype=np.uint8)
    b[:n] = mask
    return np.packbits(b.reshape(-1, 8)[:, ::-1], axis=1).ravel().view(np.uint64) if False else \
        np.packbits(b, bitorder="little").view(np.uint64)


@numba.njit(cache=True, inline="always")
def _popc(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@numba.njit(cache=True)
def _decided_pattern_counts(side, dec, k):
    """side, dec: (k, W) uint64. Counts of shots decided at all k positions, per side pattern (bit i = position i)."""
    W = side.shape[1]
    cnt = np.zeros(1 << k, dtype=np.int64)
    for w in range(W):
        m = dec[0, w]
        for i in range(1, k):
            m &= dec[i, w]
        if m == 0:
            continue
        for p in range(1 << k):
            v = m
            for i in range(k):
                if (p >> i) & 1:
                    v &= side[i, w]
                else:
                    v &= ~side[i, w]
            cnt[p] += _popc(v)
    return cnt


class FastPos:
    def __init__(self, rep: PositionRep):
        c = rep.codes
        self.rep = rep
        self.N = c.size
        self.side = _pack64(c == CODE_OMEGA1)
        self.dec = _pack64((c == CODE_OMEGA0) | (c == CODE_OMEGA1))
        self.codes = c
        self.special = np.flatnonzero((c >= 3) | (c == CODE_EXCEPTION))
        self.n_region = (int((c == CODE_OMEGA0).sum()), int((c == CODE_OMEGA1).sum()))
        self._cache: dict = {}

    def tables(self, h: Discriminator):
        """(region label sets [Omega0, Omega1], leaf label-set table) under h, cached per h."""
        key = (h.theta, h.beta)
        t = self._cache.get(key)
        if t is None:
            part = self.rep.part
            if self.rep.family.contains(h):
                reg = (LS_ONLY0, LS_ONLY1)
            else:
                reg = []
                for s in (0, 1):
                    sel = np.flatnonzero(part.status == s)
                    ls = rect_label_sets(part.x0[sel], part.y0[sel], part.x1[sel], part.y1[sel], h.theta, h.beta)
                    reg.append(int(np.bitwise_or.reduce(ls)) if sel.size else LS_ONLY0 + s)
                reg = tuple(reg)
            band = np.flatnonzero(part.status == -1)
            leaf_ls = np.zeros(part.n_leaves, dtype=np.int8)
            leaf_ls[band] = rect_label_sets(part.x0[band], part.y0[band], part.x1[band], part.y1[band], h.theta, h.beta)
            t = (reg, leaf_ls)
            self._cache[key] = t
        return t

    def shot_sets(self, idx: np.ndarray, h: Discriminator) -> np.ndarray:
        reg, leaf_ls = self.tables(h)
        c = self.codes[idx]
        out = np.empty(idx.size, dtype=np.int8)
        out[c == CODE_OMEGA0] = reg[0]
        out[c == CODE_OMEGA1] = reg[1]
        b = c >= 3
        out[b] = leaf_ls[c[b] - 3]
        e = c == CODE_EXCEPTION
        if e.any():
            ex = np.searchsorted(self.rep.exc_index, idx[e])
            out[e] = np.where(h.label(self.rep.exc_x[ex], self.rep.exc_y[ex]), LS_ONLY1, LS_ONLY0)
        return out


def pattern_intervals(fps: list[FastPos], hs: list[Discriminator]):
    """Exact intervals of every pattern count over S and of the parity sum. Returns (lo, hi, par_lo, par_hi)."""
    k = len(fps)
    regs = [fp.tables(h)[0] for fp, h in zip(fps, hs)]
    if any(r != (LS_ONLY0, LS_ONLY1) for r in regs):
        # a decided region is not a singleton under some h_j: every shot goes through the per-shot path
        idx = np.arange(fps[0].N)
    else:
        idx = np.unique(np.concatenate([fp.special for fp in fps])) if k > 1 else fps[0].special
    lo = np.zeros(1 << k, dtype=np.int64)
    if idx.size < fps[0].N:
        side = np.stack([fp.side for fp in fps])
        dec = np.stack([fp.dec for fp in fps])
        lo += _decided_pattern_counts(side, dec, k)
    hi = lo.copy()
    par_lo = par_hi = int(sum(lo[p] * (1 - 2 * (bin(p).count("1") & 1)) for p in range(1 << k)))
    if idx.size:
        ls = np.column_stack([fp.shot_sets(idx, h) for fp, h in zip(fps, hs)])
        # every shot of idx is special at some position of S, so popcount did not count it (or popcount was skipped)
        key = np.zeros(idx.size, dtype=np.int64)
        for i in range(k):
            key = key * 4 + ls[:, i]
        uk, cnt = np.unique(key, return_counts=True)
        for u, c in zip(uk.tolist(), cnt.tolist()):
            codes = []
            v = u
            for _ in range(k):
                codes.append(v % 4)
                v //= 4
            codes = codes[::-1]
            single = all(cd != LS_BOTH for cd in codes)
            if single:
                p = sum((1 << i) for i, cd in enumerate(codes) if cd == LS_ONLY1)
                lo[p] += c
                hi[p] += c
                sgn = 1 - 2 * (bin(p).count("1") & 1)
                par_lo += sgn * c
                par_hi += sgn * c
            else:
                choices = [((0,) if cd == LS_ONLY0 else (1,) if cd == LS_ONLY1 else (0, 1)) for cd in codes]
                for labs in itertools.product(*choices):
                    hi[sum(b << i for i, b in enumerate(labs))] += c
                par_lo -= c
                par_hi += c
    return lo, hi, par_lo, par_hi


# ---------------------------------------------------------------------------------------------------- one compiled call

@numba.njit(cache=True)
def _query_kernel(side, dec, codes, Sx, reg, leaf_ls, spec_ptr, spec_idx, exc_ptr, exc_idx, exc_lab, mark):
    """All statistics of one pattern-count query in one pass. side/dec: (m, W) uint64; codes: (m, N) int32; Sx: (k,)
    positions; reg: (k, 2) region label sets; leaf_ls: (k, L) leaf label sets; spec_*: CSR of special shots per position
    of the dataset; exc_*: CSR of exception shots (sorted) and their exact labels under h (per query position i);
    mark: (N,) bool scratch (all False on entry and on exit)."""
    k = Sx.shape[0]
    P = 1 << k
    lo = np.zeros(P, dtype=np.int64)
    hi = np.zeros(P, dtype=np.int64)
    par_lo = 0
    par_hi = 0
    all_single = True
    for i in range(k):
        if reg[i, 0] != 1 or reg[i, 1] != 2:
            all_single = False
    N = codes.shape[1]
    if all_single:
        W = side.shape[1]
        buf = np.empty(P, dtype=np.uint64)
        for w in range(W):
            m = dec[Sx[0], w]
            for i in range(1, k):
                m &= dec[Sx[i], w]
            if m == 0:
                continue
            # pattern-split masks: level i holds 2^(i+1) masks (2 + 4 + ... + 2^k ANDs per word instead of k 2^k)
            buf[0] = m
            nb = 1
            for i in range(k):
                sd = side[Sx[i], w]
                for q in range(nb - 1, -1, -1):
                    v = buf[q]
                    buf[2 * q + 1] = v & sd
                    buf[2 * q] = v & ~sd
                nb *= 2
            # buf index: bit (k-1-i) of the index = label at position i (built most-significant first)
            for q in range(P):
                v = buf[q]
                if v == 0:
                    continue
                pp = 0
                for i in range(k):
                    if (q >> (k - 1 - i)) & 1:
                        pp |= 1 << i
                c = _popc(v)
                lo[pp] += c
                hi[pp] += c
        for p in range(P):
            par = 1
            for i in range(k):
                if (p >> i) & 1:
                    par = -par
            par_lo += par * lo[p]
        par_hi = par_lo
    # the shots to evaluate one by one
    nlist = 0
    if all_single:
        for i in range(k):
            j = Sx[i]
            for t in range(spec_ptr[j], spec_ptr[j + 1]):
                s = spec_idx[t]
                if not mark[s]:
                    mark[s] = True
                    nlist += 1
    todo = np.empty(nlist if all_single else N, dtype=np.int64)
    if all_single:
        n2 = 0
        for i in range(k):
            j = Sx[i]
            for t in range(spec_ptr[j], spec_ptr[j + 1]):
                s = spec_idx[t]
                if mark[s]:
                    mark[s] = False
                    todo[n2] = s
                    n2 += 1
    else:
        for s in range(N):
            todo[s] = s
    for s in todo:
        can0 = 0  # bit i: label 0 possible at position i
        can1 = 0
        for i in range(k):
            j = Sx[i]
            c = codes[j, s]
            if c == 0:
                ls = reg[i, 0]
            elif c == 1:
                ls = reg[i, 1]
            elif c == 2:
                lo_ = exc_ptr[i]
                hi_ = exc_ptr[i + 1]
                a = lo_
                b = hi_
                while a < b:  # binary search in the sorted exception list of position j
                    mid = (a + b) // 2
                    if exc_idx[mid] < s:
                        a = mid + 1
                    else:
                        b = mid
                ls = 2 if exc_lab[a] else 1
            else:
                ls = leaf_ls[i, c - 3]
            if ls & 1:
                can0 |= 1 << i
            if ls & 2:
                can1 |= 1 << i
        both = can0 & can1
        if both == 0:
            p = can1
            lo[p] += 1
            hi[p] += 1
            par = 1
            for i in range(k):
                if (p >> i) & 1:
                    par = -par
            par_lo += par
            par_hi += par
        else:
            base = can1 & ~both
            sub = both
            while True:  # every subset of the ambiguous positions
                hi[base | sub] += 1
                if sub == 0:
                    break
                sub = (sub - 1) & both
            par_lo -= 1
            par_hi += 1
    return lo, hi, par_lo, par_hi


class FastDataset:
    """All positions of one dataset, decoded once for `query` (warm)."""

    def __init__(self, reps: list[PositionRep]):
        self.reps = reps
        self.fps = [FastPos(r) for r in reps]
        self.side = np.ascontiguousarray(np.stack([f.side for f in self.fps]))
        self.dec = np.ascontiguousarray(np.stack([f.dec for f in self.fps]))
        self.codes = np.ascontiguousarray(np.stack([r.codes for r in reps]).astype(np.int32))
        sp = [f.special for f in self.fps]
        self.spec_ptr = np.concatenate([[0], np.cumsum([a.size for a in sp])]).astype(np.int64)
        self.spec_idx = np.concatenate(sp).astype(np.int64) if sp else np.zeros(0, np.int64)
        self.mark = np.zeros(reps[0].codes.size, dtype=np.bool_)
        self.L = max(r.part.n_leaves for r in reps)
        self._exc_cache: dict = {}

    def _exc_labels(self, j: int, h: Discriminator) -> np.ndarray:
        key = (j, h.theta, h.beta)
        v = self._exc_cache.get(key)
        if v is None:
            r = self.reps[j]
            v = h.label(r.exc_x, r.exc_y) if r.exc_index.size else np.zeros(0, dtype=bool)
            self._exc_cache[key] = v
        return v

    def query(self, S: list[int], hs: list[Discriminator]):
        k = len(S)
        reg = np.empty((k, 2), dtype=np.int64)
        leaf = np.zeros((k, self.L), dtype=np.int8)
        exc_ptr = np.zeros(k + 1, dtype=np.int64)
        exc_idx, exc_lab = [], []
        for i, (j, h) in enumerate(zip(S, hs)):
            rg, lt = self.fps[j].tables(h)
            reg[i] = rg
            leaf[i, :lt.size] = lt
            r = self.reps[j]
            exc_ptr[i + 1] = exc_ptr[i] + r.exc_index.size
            exc_idx.append(r.exc_index)
            exc_lab.append(self._exc_labels(j, h))
        ei = np.concatenate(exc_idx).astype(np.int64) if exc_ptr[-1] else np.zeros(1, np.int64)
        el = np.concatenate(exc_lab).astype(np.bool_) if exc_ptr[-1] else np.zeros(1, np.bool_)
        return _query_kernel(self.side, self.dec, self.codes, np.asarray(S, dtype=np.int64), reg, leaf, self.spec_ptr,
                             self.spec_idx, exc_ptr, ei, el, self.mark)
