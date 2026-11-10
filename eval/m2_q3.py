"""Q3 / H-c: view maintenance under discriminator version changes (prereg/M2.md v4 section 5 Q3; Theorem 3 of the paper).

Per dataset (resident decoded representation, built once at load, identically for both methods):
  views   classes 1-3 of section 4 (every position; adjacent pairs; 50 random 3- and 4-subsets), each a vector of exact
          pattern-count intervals (lo, hi over 2^k patterns) and the parity interval, under the current version;
  tables  per position, the label set of every stored cell under the current discriminator (code 0 / 1: the decided
          regions, code 3 + leaf: leaves) and the exact label of every exception.
An event applies a new discriminator vector at the changed positions J (one batch per dataset).
  IQSTORE (incremental, Theorem 3): new tables for J; the cells whose label set changed; their shots from the cell ->
          shots index (CSR, built at load); every view touching J subtracts the old and adds the new terms of exactly
          those shots.
  B6 (full recompute from the same representation): new tables for J; every view recomputed with the word-level kernel
          of the query path (`fastq._query_kernel`: popcount over decided shots + per-shot evaluation of the others).
Both methods run their per-event work in one compiled call each, after the same table evaluation, so the comparison
is between maintenance algorithms on the same engine. Correctness: after every event the incremental views equal the
recomputed ones exactly (integer arithmetic).
"""
from __future__ import annotations

import time

import numba
import numpy as np

from iqstore.family import Discriminator, unit
from iqstore.fastq import _query_kernel, _pack64
from iqstore.rep import CODE_EXCEPTION, CODE_OMEGA0, CODE_OMEGA1, LS_ONLY0, LS_ONLY1, Partition, rect_label_sets

KMAX = 4
_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)


# ------------------------------------------------------------------------------------------------ static structures

class Store:
    """Resident decoded representation of one dataset plus the maintenance structures (built at load)."""

    def __init__(self, reps):
        self.reps = reps
        self.m = len(reps)
        self.N = reps[0].codes.size
        N, m = self.N, self.m
        self.codes = np.ascontiguousarray(np.stack([r.codes for r in reps]).astype(np.int32))
        self.T = max(r.part.n_leaves for r in reps) + 3  # table size: codes 0, 1, 2 and 3 + leaf
        # cell -> shots CSR per position (code-sorted shot list)
        self.csr_ptr = np.zeros((m, self.T + 1), np.int64)
        self.csr_idx = np.zeros((m, N), np.int64)
        for j in range(m):
            c = self.codes[j]
            order = np.argsort(c, kind="stable")
            self.csr_idx[j] = order
            cnt = np.bincount(c, minlength=self.T)
            self.csr_ptr[j, 1:] = np.cumsum(cnt)
        # query-path structures for B6 (static: decided membership and specials do not depend on h)
        self.side = np.ascontiguousarray(np.stack([_pack64(r.codes == CODE_OMEGA1) for r in reps]))
        self.dec = np.ascontiguousarray(np.stack([_pack64((r.codes == CODE_OMEGA0) | (r.codes == CODE_OMEGA1))
                                                  for r in reps]))
        sp = [np.flatnonzero((r.codes >= 3) | (r.codes == CODE_EXCEPTION)) for r in reps]
        self.spec_ptr = np.concatenate([[0], np.cumsum([a.size for a in sp])]).astype(np.int64)
        self.spec_idx = np.concatenate(sp).astype(np.int64) if self.spec_ptr[-1] else np.zeros(1, np.int64)
        ex = [r.exc_index.astype(np.int64) for r in reps]
        self.exc_ptr = np.concatenate([[0], np.cumsum([a.size for a in ex])]).astype(np.int64)
        self.exc_idx = np.concatenate(ex) if self.exc_ptr[-1] else np.zeros(1, np.int64)
        self.occupied = [np.flatnonzero(np.bincount(r.codes[r.codes >= 3] - 3, minlength=r.part.n_leaves) > 0)
                         for r in reps]
        self.decided_leaves = [np.flatnonzero(r.part.status >= 0) for r in reps]
        # flattened per-position leaf geometry and leaf lists for the batched table kernel
        self.roots = np.array([r.part.root for r in reps], dtype=np.float64)
        self.lptr = np.concatenate([[0], np.cumsum([r.part.n_leaves for r in reps])]).astype(np.int64)
        self.lD = np.concatenate([r.part.depth for r in reps]).astype(np.int64)
        self.lI = np.concatenate([r.part.ix for r in reps]).astype(np.int64)
        self.lJ = np.concatenate([r.part.iy for r in reps]).astype(np.int64)
        self.lS = np.concatenate([r.part.status for r in reps]).astype(np.int8)
        self.optr = np.concatenate([[0], np.cumsum([a.size for a in self.occupied])]).astype(np.int64)
        self.oidx = np.concatenate(self.occupied).astype(np.int64) if self.optr[-1] else np.zeros(1, np.int64)
        self.dptr = np.concatenate([[0], np.cumsum([a.size for a in self.decided_leaves])]).astype(np.int64)
        self.didx = np.concatenate(self.decided_leaves).astype(np.int64) if self.dptr[-1] else np.zeros(1, np.int64)
        # nonempty cells (codes holding at least one shot) per position
        ne = [np.flatnonzero(np.diff(self.csr_ptr[j]) > 0) for j in range(m)]
        self.cptr = np.concatenate([[0], np.cumsum([a.size for a in ne])]).astype(np.int64)
        self.cidx = np.concatenate(ne).astype(np.int64)
        self.has_exc = [r.exc_index.size > 0 for r in reps]
        # shot -> slot in its position's exception list (static; shared by every Maintainer of this store)
        self.E = max(int(r.exc_index.size) for r in reps)
        self.exc_pos = np.zeros((m, N), np.int32)
        for j, r in enumerate(reps):
            self.exc_pos[j, r.exc_index] = np.arange(r.exc_index.size)
        # shot-major copies for the per-shot view updates of `incremental`
        self.codesT = np.ascontiguousarray(self.codes.T)
        self.excT = np.ascontiguousarray(self.exc_pos.T)

    def tables(self, j: int, h: Discriminator) -> tuple[np.ndarray, np.ndarray]:
        """(label set per code of position j under h, exact labels of its exceptions). Leaves: the occupied boundary
        leaves (other leaves hold no shot); regions: {0} / {1} inside the family, else the union over their leaves.
        Compiled float filter with exact rational fallback (`rep.rect_label_sets`) for the comparisons it cannot settle."""
        r = self.reps[j]
        part = r.part
        tab = np.zeros(self.T, np.int8)
        pp, qq = unit(h.theta)
        in_fam = r.family.contains(h)
        occ, dl = self.occupied[j], self.decided_leaves[j]
        unsure = np.empty(occ.size + dl.size, np.int64)
        nu = _tables_kernel(float(part.root[0]), float(part.root[1]), float(part.root[2]), part.depth, part.ix, part.iy,
                            part.status, occ, dl, pp, qq, float(h.beta), in_fam, tab, unsure)
        if nu:
            u = unsure[:nu]
            sub = Partition(part.root, part.ix[u], part.iy[u], part.depth[u], part.status[u])
            ls = rect_label_sets(sub.x0, sub.y0, sub.x1, sub.y1, h.theta, h.beta)
            for leaf, v in zip(u.tolist(), ls.tolist()):
                if part.status[leaf] == -1:
                    tab[3 + leaf] = v
                else:
                    tab[part.status[leaf]] |= v
            if not in_fam:
                for s in (0, 1):
                    if tab[s] == 0:
                        tab[s] = LS_ONLY0 + s
        lab = h.label(r.exc_x, r.exc_y) if r.exc_index.size else np.zeros(0, bool)
        return tab, lab


@numba.njit(cache=True)
def _sign_filter(x, y, p, q, b):
    px = p * x
    qy = q * y
    d = (px + qy) - b
    err = 2.0 * _EPS * (abs(px) + abs(qy) + abs(b)) + _TINY
    if not (abs(d) > err):
        return 2
    return 1 if d > 0 else (-1 if d < 0 else 0)


@numba.njit(cache=True)
def _leaf_ls(X0, Y0, side, d, i, j, p, q, beta):
    c = side / (2.0 ** d)
    x0 = X0 + i * c
    x1 = X0 + (i + 1) * c
    y0 = Y0 + j * c
    y1 = Y0 + (j + 1) * c
    xu = x1 if p > 0 else x0
    yu = y1 if q > 0 else y0
    xl = x0 if p >= 0 else x1
    yl = y0 if q >= 0 else y1
    su = _sign_filter(xu, yu, p, q, beta)
    sl = _sign_filter(xl, yl, p, q, beta)
    if su == 2 or sl == 2:
        return -1
    v = 0
    if sl < 0 or (sl == 0 and p >= 0 and q >= 0):
        v |= 1
    if su > 0:
        v |= 2
    return v


@numba.njit(cache=True)
def _tables_kernel(X0, Y0, side, D, I, J, S, occ, dl, p, q, beta, in_fam, tab, unsure):
    """`rep.rect_label_sets` of the occupied boundary leaves and (outside the family) of the decided leaves, with the
    float filter of `family.exact_sign`; unsettled leaves are listed in `unsure`. Returns their number."""
    nu = 0
    for a in range(occ.size):
        k = occ[a]
        v = _leaf_ls(X0, Y0, side, D[k], I[k], J[k], p, q, beta)
        if v < 0:
            unsure[nu] = k
            nu += 1
        else:
            tab[3 + k] = v
    if in_fam:
        tab[0] = 1
        tab[1] = 2
        return nu
    r0 = 0
    r1 = 0
    for a in range(dl.size):
        k = dl[a]
        v = _leaf_ls(X0, Y0, side, D[k], I[k], J[k], p, q, beta)
        if v < 0:
            unsure[nu] = k
            nu += 1
        elif S[k] == 0:
            r0 |= v
        else:
            r1 |= v
    tab[0] = r0
    tab[1] = r1
    if nu == 0:
        if tab[0] == 0:
            tab[0] = 1
        if tab[1] == 0:
            tab[1] = 2
    return nu


@numba.njit(cache=True)
def _tables_batch(J, roots, lptr, lD, lI, lJ, lS, optr, oidx, dptr, didx, pqb, infam, tab, unsure_pos, unsure_leaf):
    """`_tables_kernel` for every changed position in one call; unsettled leaves are returned as (position, leaf)."""
    nu = 0
    for a in range(J.size):
        j = J[a]
        X0, Y0, side = roots[j, 0], roots[j, 1], roots[j, 2]
        p, q, beta = pqb[a, 0], pqb[a, 1], pqb[a, 2]
        base = lptr[j]
        for t in range(optr[j], optr[j + 1]):
            k = oidx[t]
            v = _leaf_ls(X0, Y0, side, lD[base + k], lI[base + k], lJ[base + k], p, q, beta)
            if v < 0:
                unsure_pos[nu] = j
                unsure_leaf[nu] = k
                nu += 1
            else:
                tab[j, 3 + k] = v
        if infam[a]:
            tab[j, 0] = 1
            tab[j, 1] = 2
            continue
        r0 = 0
        r1 = 0
        for t in range(dptr[j], dptr[j + 1]):
            k = didx[t]
            v = _leaf_ls(X0, Y0, side, lD[base + k], lI[base + k], lJ[base + k], p, q, beta)
            if v < 0:
                unsure_pos[nu] = j
                unsure_leaf[nu] = k
                nu += 1
            elif lS[base + k] == 0:
                r0 |= v
            else:
                r1 |= v
        tab[j, 0] = r0 if r0 != 0 else 1
        tab[j, 1] = r1 if r1 != 0 else 2
    return nu


class Views:
    def __init__(self, sets: list[list[int]]):
        V = len(sets)
        self.S = np.full((V, KMAX), -1, np.int64)
        self.k = np.zeros(V, np.int64)
        for v, s in enumerate(sets):
            self.S[v, :len(s)] = s
            self.k[v] = len(s)
        self.lo = np.zeros((V, 1 << KMAX), np.int64)
        self.hi = np.zeros((V, 1 << KMAX), np.int64)
        self.par = np.zeros((V, 2), np.int64)

    def copy_state(self):
        return self.lo.copy(), self.hi.copy(), self.par.copy()


def views_touching(views: Views, m: int) -> tuple[np.ndarray, np.ndarray]:
    lists = [[] for _ in range(m)]
    for v in range(views.S.shape[0]):
        for i in range(views.k[v]):
            lists[views.S[v, i]].append(v)
    ptr = np.concatenate([[0], np.cumsum([len(a) for a in lists])]).astype(np.int64)
    idx = np.array([v for a in lists for v in a], dtype=np.int64) if ptr[-1] else np.zeros(1, np.int64)
    return ptr, idx


# ------------------------------------------------------------------------------------------------ compiled kernels

@numba.njit(cache=True)
def _ls(j, s, codes, tab, exl, exc_pos):
    c = codes[j, s]
    if c == 2:
        return exl[j, exc_pos[j, s]]
    return tab[j, c]


@numba.njit(cache=True)
def _add_term(lo, hi, par, v, k, lsv, sgn):
    can0 = 0
    can1 = 0
    for i in range(k):
        if lsv[i] & 1:
            can0 |= 1 << i
        if lsv[i] & 2:
            can1 |= 1 << i
    both = can0 & can1
    if both == 0:
        p = can1
        lo[v, p] += sgn
        hi[v, p] += sgn
        pr = 1
        for i in range(k):
            if (p >> i) & 1:
                pr = -pr
        par[v, 0] += sgn * pr
        par[v, 1] += sgn * pr
    else:
        base = can1 & ~both
        sub = both
        while True:
            hi[v, base | sub] += sgn
            if sub == 0:
                break
            sub = (sub - 1) & both
        par[v, 0] -= sgn
        par[v, 1] += sgn


@numba.njit(cache=True)
def _term(lo, hi, par, v, k, can0, can1, sgn):
    """`_add_term` from the masks of positions whose label set contains 0 (can0) and 1 (can1)."""
    both = can0 & can1
    if both == 0:
        p = can1
        lo[v, p] += sgn
        hi[v, p] += sgn
        pr = 1
        for i in range(k):
            if (p >> i) & 1:
                pr = -pr
        par[v, 0] += sgn * pr
        par[v, 1] += sgn * pr
    else:
        base = can1 & ~both
        sub = both
        while True:
            hi[v, base | sub] += sgn
            if sub == 0:
                break
            sub = (sub - 1) & both
        par[v, 0] -= sgn
        par[v, 1] += sgn


@numba.njit(cache=True)
def incremental(codes, tab_old, tab_new, exl_old, exl_new, exc_pos, J, csr_ptr, csr_idx, cptr, cidx, vS, vk, vptr,
                vidx, lo, hi, par, mark, vmark, redo, buf, codesT, excT):
    """Theorem 3: apply the version change at positions J (tables already evaluated). Per changed position, the shots
    of the cells whose label set changed (and the exceptions whose label changed); per view touching J, the union of
    those lists over its changed positions gets its old terms subtracted and its new terms added. A view whose list
    exceeds N/8 shots is flagged in `redo` for a recompute instead (cheaper for wide changes, e.g. a decided region whose
    label set changes outside the family). The per-shot loop reads the shot-major copies `codesT` (N x m) and `excT`
(N x m) so that the k codes of one shot are adjacent in memory, and skips a shot whose label-set masks on the view are
unchanged. Returns (cells changed, shots touched, view-shot updates, views redone)."""
    m, N = codes.shape
    T = tab_old.shape[1]
    nJ = J.size
    # per changed position: affected shots (CSR over J)
    aptr = np.zeros(nJ + 1, np.int64)
    n_cells = 0
    for a in range(nJ):
        j = J[a]
        c_ = 0
        for cc in range(cptr[j], cptr[j + 1]):
            c = cidx[cc]
            if c == 2:
                for q in range(csr_ptr[j, 2], csr_ptr[j, 3]):
                    s = csr_idx[j, q]
                    e = exc_pos[j, s]
                    if exl_old[j, e] != exl_new[j, e]:
                        c_ += 1
            elif tab_old[j, c] != tab_new[j, c]:
                n_cells += 1
                c_ += csr_ptr[j, c + 1] - csr_ptr[j, c]
        aptr[a + 1] = aptr[a] + c_
    aidx = np.empty(aptr[nJ], np.int64)
    for a in range(nJ):
        j = J[a]
        t = aptr[a]
        for cc in range(cptr[j], cptr[j + 1]):
            c = cidx[cc]
            if c == 2:
                for q in range(csr_ptr[j, 2], csr_ptr[j, 3]):
                    s = csr_idx[j, q]
                    e = exc_pos[j, s]
                    if exl_old[j, e] != exl_new[j, e]:
                        aidx[t] = s
                        t += 1
            elif tab_old[j, c] != tab_new[j, c]:
                for q in range(csr_ptr[j, c], csr_ptr[j, c + 1]):
                    aidx[t] = csr_idx[j, q]
                    t += 1
    slot = np.full(m, -1, np.int64)
    for a in range(nJ):
        slot[J[a]] = a
    # views touching J
    nv = 0
    vlist = np.empty(max(vidx.size, 1), np.int64)
    for a in range(nJ):
        j = J[a]
        for q in range(vptr[j], vptr[j + 1]):
            v = vidx[q]
            if not vmark[v]:
                vmark[v] = True
                vlist[nv] = v
                nv += 1
    upd = 0
    nredo = 0
    touched_total = 0
    for b in range(nv):
        v = vlist[b]
        vmark[v] = False
        k = vk[v]
        nb = 0
        for i in range(k):
            a = slot[vS[v, i]]
            if a < 0:
                continue
            for q in range(aptr[a], aptr[a + 1]):
                s = aidx[q]
                if not mark[s]:
                    mark[s] = True
                    buf[nb] = s
                    nb += 1
        for q in range(nb):
            mark[buf[q]] = False
        if nb > N // 8:
            redo[v] = True
            nredo += 1
            continue
        touched_total += nb
        for q in range(nb):
            s = buf[q]
            o0 = 0
            o1 = 0
            n0 = 0
            n1 = 0
            for i in range(k):
                j = vS[v, i]
                c = codesT[s, j]
                if c == 2:
                    e = excT[s, j]
                    a_ = exl_old[j, e]
                    b_ = exl_new[j, e]
                else:
                    a_ = tab_old[j, c]
                    b_ = tab_new[j, c]
                if a_ & 1:
                    o0 |= 1 << i
                if a_ & 2:
                    o1 |= 1 << i
                if b_ & 1:
                    n0 |= 1 << i
                if b_ & 2:
                    n1 |= 1 << i
            if o0 == n0 and o1 == n1:
                continue
            _term(lo, hi, par, v, k, o0, o1, -1)
            _term(lo, hi, par, v, k, n0, n1, 1)
            upd += 1
    return n_cells, aptr[nJ], upd, nredo


@numba.njit(cache=True)
def recompute(side, dec, codes, tab, exl_flat, spec_ptr, spec_idx, exc_ptr, exc_idx, vS, vk, lo, hi, par, mark, sel):
    """Recompute the selected views from the representation with the word-level query kernel (B6: all views)."""
    V = vS.shape[0]
    T = tab.shape[1]
    for v in range(V):
        if not sel[v]:
            continue
        k = vk[v]
        Sx = vS[v, :k].copy()
        reg = np.empty((k, 2), np.int64)
        leaf = np.zeros((k, T - 3 if T > 3 else 1), np.int8)
        for i in range(k):
            j = Sx[i]
            reg[i, 0] = tab[j, 0]
            reg[i, 1] = tab[j, 1]
            for c in range(3, T):
                leaf[i, c - 3] = tab[j, c]
        e_ptr = np.zeros(k + 1, np.int64)
        for i in range(k):
            j = Sx[i]
            e_ptr[i + 1] = e_ptr[i] + exc_ptr[j + 1] - exc_ptr[j]
        e_idx = np.zeros(max(e_ptr[k], 1), np.int64)
        e_lab = np.zeros(max(e_ptr[k], 1), np.bool_)
        for i in range(k):
            j = Sx[i]
            for q in range(exc_ptr[j], exc_ptr[j + 1]):
                e_idx[e_ptr[i] + q - exc_ptr[j]] = exc_idx[q]
                e_lab[e_ptr[i] + q - exc_ptr[j]] = exl_flat[q] == 2
        a, b, c0, c1 = _query_kernel(side, dec, codes, Sx, reg, leaf, spec_ptr, spec_idx, e_ptr, e_idx, e_lab, mark)
        P = 1 << k
        for p in range(P):
            lo[v, p] = a[p]
            hi[v, p] = b[p]
        par[v, 0] = c0
        par[v, 1] = c1


# ------------------------------------------------------------------------------------------------ driver

class Maintainer:
    """Holds the current version and the views of one dataset for both methods."""

    def __init__(self, store: Store, view_sets: list[list[int]], h0: list[Discriminator]):
        self.st = store
        m, N = store.m, store.N
        self.views_iq = Views(view_sets)
        self.views_b6 = Views(view_sets)
        self.vptr, self.vidx = views_touching(self.views_iq, m)
        self.h = list(h0)
        E = store.E
        self.exc_pos = store.exc_pos
        self.tab = np.zeros((m, store.T), np.int8)
        self.exl = np.ones((m, max(E, 1)), np.int8)
        for j in range(m):
            t, lab = store.tables(j, self.h[j])
            self.tab[j] = t
            self.exl[j, :lab.size] = np.where(lab, LS_ONLY1, LS_ONLY0)
        self.mark = np.zeros(N, np.bool_)
        self.buf = np.empty(N, np.int64)
        self.vmark = np.zeros(self.views_iq.S.shape[0], np.bool_)
        self._recompute(self.views_iq, self.tab, self.exl)
        self._recompute(self.views_b6, self.tab, self.exl)

    def _exl_flat(self, exl):
        st = self.st
        out = np.zeros(max(int(st.exc_ptr[-1]), 1), np.int8)
        for j in range(st.m):
            n = int(st.exc_ptr[j + 1] - st.exc_ptr[j])
            out[st.exc_ptr[j]:st.exc_ptr[j] + n] = exl[j, :n]
        return out

    def _recompute(self, views, tab, exl, sel=None):
        st = self.st
        if sel is None:
            sel = np.ones(views.S.shape[0], np.bool_)
        recompute(st.side, st.dec, st.codes, tab, self._exl_flat(exl), st.spec_ptr, st.spec_idx, st.exc_ptr,
                  st.exc_idx, views.S, views.k, views.lo, views.hi, views.par, self.mark, sel)

    def new_tables(self, J, hs_new):
        """Label-set tables of the changed positions under the new discriminators (one compiled call; exact rational
        fallback for unsettled comparisons; exact labels of exceptions)."""
        st = self.st
        tab, exl = self.tab.copy(), self.exl.copy()
        pqb = np.empty((len(J), 3), np.float64)
        infam = np.empty(len(J), np.bool_)
        for a, j in enumerate(J.tolist()):
            h = hs_new[j]
            pqb[a, 0], pqb[a, 1] = unit(h.theta)
            pqb[a, 2] = h.beta
            infam[a] = st.reps[j].family.contains(h)
        cap = int(st.optr[-1] + st.dptr[-1]) + 1
        upos = np.empty(cap, np.int64)
        uleaf = np.empty(cap, np.int64)
        nu = _tables_batch(J, st.roots, st.lptr, st.lD, st.lI, st.lJ, st.lS, st.optr, st.oidx, st.dptr, st.didx, pqb,
                           infam, tab, upos, uleaf)
        if nu:  # exact resolution, then the region unions again for the affected positions
            redo_pos = sorted(set(upos[:nu].tolist()))
            for j in redo_pos:
                tab[j], _ = st.tables(j, hs_new[j])
        for j in J.tolist():
            if st.has_exc[j]:
                r = st.reps[j]
                lab = hs_new[j].label(r.exc_x, r.exc_y)
                exl[j, :lab.size] = np.where(lab, LS_ONLY1, LS_ONLY0)
        return tab, exl

    def _run_iq(self, Ja, hs_new):
        st = self.st
        redo = np.zeros(self.views_iq.S.shape[0], np.bool_)
        t0 = time.perf_counter()
        tab, exl = self.new_tables(Ja, hs_new)
        t1 = time.perf_counter()
        n = incremental(st.codes, self.tab, tab, self.exl, exl, self.exc_pos, Ja, st.csr_ptr, st.csr_idx, st.cptr,
                        st.cidx, self.views_iq.S, self.views_iq.k, self.vptr, self.vidx, self.views_iq.lo,
                        self.views_iq.hi, self.views_iq.par, self.mark, self.vmark, redo, self.buf,
                        st.codesT, st.excT)
        if n[3]:
            self._recompute(self.views_iq, tab, exl, redo)
        t2 = time.perf_counter()
        return tab, exl, n, t2 - t0, t1 - t0

    def _run_b6(self, Ja, hs_new):
        t0 = time.perf_counter()
        tab, exl = self.new_tables(Ja, hs_new)
        self._recompute(self.views_b6, tab, exl)
        return time.perf_counter() - t0

    def _equal(self) -> bool:
        a, b = self.views_iq, self.views_b6
        return np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi) and np.array_equal(a.par, b.par)

    def event_timed(self, J: list[int], hs_new: list[Discriminator], order: np.ndarray, a6=None) -> dict:
        """prereg section 6 runs for one maintenance event: every run starts from the pre-event state (restored untimed);
        `order[r]` is the permutation of the methods in run r (0 = IQSTORE, 1 = B6, 2 = A6 if given); the first 2 runs
        are warm-ups. Equality of the incremental views with the full recompute is checked after EVERY run; A6's exact
        views are checked once per event (exact recount from the raw points, and containment in IQSTORE's intervals).
        The state advances once, at the end."""
        Ja = np.array(sorted(J), dtype=np.int64)
        snap_iq = self.views_iq.copy_state()
        snap_b6 = self.views_b6.copy_state()
        snap_a6 = a6.snapshot() if a6 is not None else None
        t_iq, t_b6, t_tab, t_a6 = [], [], [], []
        n = w6 = None
        for r in range(order.shape[0]):
            for which in order[r].tolist():
                if which == 0:
                    self.views_iq.lo[:], self.views_iq.hi[:], self.views_iq.par[:] = snap_iq
                    tab, exl, n, dt, dtab = self._run_iq(Ja, hs_new)
                    t_iq.append(dt)
                    t_tab.append(dtab)
                elif which == 1:
                    self.views_b6.lo[:], self.views_b6.hi[:], self.views_b6.par[:] = snap_b6
                    t_b6.append(self._run_b6(Ja, hs_new))
                elif a6 is not None:
                    a6.restore(snap_a6)
                    dt, w6 = a6.event(sorted(J), hs_new)
                    t_a6.append(dt)
            if not self._equal():
                raise AssertionError("incremental views differ from the full recompute")
        self.tab, self.exl = tab, exl
        for j in J:
            self.h[j] = hs_new[j]
        if a6 is not None:
            a6.check(self.views_iq)
        out = {"t_iq": t_iq, "t_b6": t_b6, "t_tables": t_tab, "cells": int(n[0]), "shots": int(n[1]),
               "view_shot_updates": int(n[2]), "views_recomputed": int(n[3])}
        if a6 is not None:
            out |= {"t_a6": t_a6, "a6_work": w6}
        return out

    def reset(self, hs: list[Discriminator]) -> None:
        """Untimed reset to the version `hs` (every position): tables and both view sets recomputed."""
        J = np.arange(self.st.m, dtype=np.int64)
        tab, exl = self.new_tables(J, hs)
        self._recompute(self.views_iq, tab, exl)
        self._recompute(self.views_b6, tab, exl)
        self.tab, self.exl, self.h = tab, exl, list(hs)

    def event(self, J: list[int], hs_new: list[Discriminator]) -> dict:
        """Apply the version change at J with both methods, timed separately; checks equality."""
        st = self.st
        Ja = np.array(sorted(J), dtype=np.int64)
        redo = np.zeros(self.views_iq.S.shape[0], np.bool_)
        t0 = time.perf_counter()
        tab, exl = self.new_tables(Ja, hs_new)
        t1 = time.perf_counter()
        n_cells, n_shots, n_upd, n_redo = incremental(st.codes, self.tab, tab, self.exl, exl, self.exc_pos, Ja,
                                                      st.csr_ptr, st.csr_idx, st.cptr, st.cidx, self.views_iq.S,
                                                      self.views_iq.k, self.vptr, self.vidx, self.views_iq.lo,
                                                      self.views_iq.hi, self.views_iq.par, self.mark, self.vmark, redo,
                                                      self.buf, st.codesT, st.excT)
        if n_redo:
            self._recompute(self.views_iq, tab, exl, redo)
        t2 = time.perf_counter()
        tab2, exl2 = self.new_tables(Ja, hs_new)
        t3 = time.perf_counter()
        self._recompute(self.views_b6, tab2, exl2)
        t4 = time.perf_counter()
        ok = (np.array_equal(self.views_iq.lo, self.views_b6.lo) and np.array_equal(self.views_iq.hi, self.views_b6.hi)
              and np.array_equal(self.views_iq.par, self.views_b6.par))
        if not ok:
            raise AssertionError("incremental views differ from the full recompute")
        self.tab, self.exl = tab, exl
        for j in J:
            self.h[j] = hs_new[j]
        return {"t_iq": (t1 - t0) + (t2 - t1), "t_b6": (t3 - t2) + (t4 - t3), "t_tables": t1 - t0,
                "cells": int(n_cells), "shots": int(n_shots), "view_shot_updates": int(n_upd),
                "views_recomputed": int(n_redo)}
