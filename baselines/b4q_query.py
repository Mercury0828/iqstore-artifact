"""B4q end-to-end query engine on the same infrastructure as IQSTORE's timed engine (`iqstore/lean.py`): one read
(`storage.read_file`), compiled container parse, the same pattern-count kernel, the same |S| = 1 counting. Label-set
rule = `eval.m2_cov.b4q_label_sets`: outside the slab the stored side bit for in-family h, {0, 1} otherwise; slab shots the exact half-open label set of their
grid cell [kx d, (kx+1) d) x [ky d, (ky+1) d); B4 fallback (kind 4): exact labels of the stored raw slab points."""
from __future__ import annotations

import struct

import numba
import numpy as np

from iqstore.e2e import _REV, _expand_sparse, _query_kernel_sp, _sign_filter, _zload
from iqstore.family import Family, unit
from iqstore.frame import Frame
from iqstore.lean import _true_pos_b, _u32
from iqstore.rep import rect_label_sets
from iqstore.storage import DTYPES, read_file

_B4H = struct.Struct("<B4d2d2dd4dqB")
_HEAD = struct.Struct("<qqI")


@numba.njit(cache=True)
def _groups(buf, pos, m, S):
    k = S.size
    off = np.zeros((k, 16), np.int64)
    ln = np.zeros((k, 16), np.int64)
    cnt = np.zeros(k, np.int64)
    slot = np.full(m, -1, np.int64)
    last = -1
    for i in range(k):
        slot[S[i]] = i
        if S[i] > last:
            last = S[i]
    for g in range(last + 1):
        L = _u32(buf, pos)
        n = buf[pos + 4] | (buf[pos + 5] << 8)
        pos += 4 + L
        s = slot[g]
        for b in range(n):
            L = _u32(buf, pos)
            if s >= 0 and b < 16:
                off[s, b] = pos + 4
                ln[s, b] = L
            pos += 4 + L
        if s >= 0:
            cnt[s] = n
    return off, ln, cnt


@numba.njit(cache=True)
def _rect_ls(x0, y0, x1, y1, p, q, beta):
    """`rep.rect_label_sets` of one half-open rectangle with the float filter (same corner choice as
    `e2e._leaf_label_set`); -1 when the filter cannot settle it."""
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
def _cell_ls(uk, d, p, q, beta, out, unsure):
    nu = 0
    for c in range(uk.shape[0]):
        kx = uk[c, 0]
        ky = uk[c, 1]
        v = _rect_ls(kx * d, ky * d, (kx + 1) * d, (ky + 1) * d, p, q, beta)
        if v < 0:
            unsure[nu] = c
            nu += 1
        else:
            out[c] = v
    return nu


@numba.njit(cache=True)
def _b4q_single(N, slab_pos, sideb, inv, cell_ls, exact_lab, infam):
    """|S| = 1 counts: (c0, c1, c2) = shots with label set {0}, {1}, {0, 1}."""
    n_slab = slab_pos.size
    n_out = N - n_slab
    if infam:
        c1 = 0
        for t in range(sideb.size):
            v = sideb[t]
            while v:
                v &= v - 1
                c1 += 1
        c0 = n_out - c1
        c2 = 0
    else:
        c0 = 0
        c1 = 0
        c2 = n_out
    if exact_lab.size == n_slab and inv.size == 0:
        for t in range(n_slab):
            if exact_lab[t]:
                c1 += 1
            else:
                c0 += 1
        return c0, c1, c2
    w = np.zeros(cell_ls.size, np.int64)
    for t in range(inv.size):
        w[inv[t]] += 1
    for c in range(cell_ls.size):
        v = cell_ls[c]
        if v == 1:
            c0 += w[c]
        elif v == 2:
            c1 += w[c]
        else:
            c2 += w[c]
    return c0, c1, c2


class B4qFile:
    """B4q engine over a persisted B4q container (one file per dataset; groups = positions in order)."""

    def __init__(self, path):
        self.path = path

    def query(self, S, hs):
        data = read_file(self.path)
        buf = np.frombuffer(data, dtype=np.uint8)
        ln = data[0]
        N, m, nb = _HEAD.unpack_from(data, 1 + ln)
        off, lens, cnt = _groups(buf, 1 + ln + 20, m, np.array(S, dtype=np.int64))
        mv = memoryview(data)
        k = len(S)
        per = []
        for i in range(k):
            o = off[i].tolist()
            L = lens[i].tolist()
            hd = _B4H.unpack_from(data, o[0])
            kind = hd[0]
            fam = Family(*hd[10:14])
            Nn = hd[14]
            h = hs[i]
            infam = fam.contains(h)
            mb = mv[o[1]:o[1] + L[1]]
            if mb[0] == 66:
                pay = np.frombuffer(_zload(mb[1:]), dtype=np.uint8)
                slab_pos = _true_pos_b(pay, 0, pay.size, 0, Nn)
            else:
                gp = np.frombuffer(_zload(mb[2:]), dtype=np.uint8)
                isz = np.dtype(DTYPES[mb[1]]).itemsize
                slab_pos = _true_pos_b(gp, 0, gp.size, isz, Nn)
            sideb = np.frombuffer(_zload(mv[o[2]:o[2] + L[2]]), dtype=np.uint8)
            if kind == 5:
                cb = mv[o[3]:o[3] + L[3]]
                ncell = struct.unpack_from("<q", cb, 0)[0]
                uk = np.frombuffer(_zload(cb[8:]), dtype=np.int32).reshape(ncell, 2).astype(np.int64)
                inv = np.frombuffer(_zload(mv[o[4]:o[4] + L[4]]), dtype=DTYPES[hd[15]]).astype(np.int64)
                pp, qq = unit(h.theta)
                cls = np.zeros(ncell, np.int8)
                unsure = np.empty(ncell, np.int64)
                nu = _cell_ls(uk, hd[9], pp, qq, h.beta, cls, unsure)
                if nu:
                    u = unsure[:nu]
                    d = hd[9]
                    cls[u] = rect_label_sets(uk[u, 0] * d, uk[u, 1] * d, (uk[u, 0] + 1) * d, (uk[u, 1] + 1) * d, h.theta,
                                             h.beta)
                lab = np.zeros(0, np.bool_)
            else:  # B4 fallback: exact slab points
                rb = mv[o[3]:o[3] + L[3]]
                fr = Frame.from_array(np.array(hd[1:5]))
                raw = np.frombuffer(_zload(rb), dtype=DTYPES[hd[15]])
                ex, ey = fr.apply(raw)
                lab = np.ascontiguousarray(h.label(ex, ey), dtype=np.bool_)
                uk = np.zeros((0, 2), np.int64)
                inv = np.zeros(0, np.int64)
                cls = np.zeros(0, np.int8)
            per.append((Nn, slab_pos, sideb, inv, cls, lab, infam, kind))
        if k == 1:
            Nn, slab_pos, sideb, inv, cls, lab, infam, kind = per[0]
            c0, c1, c2 = _b4q_single(Nn, slab_pos, sideb, inv, cls, lab, infam)
            return (np.array([c0, c1], np.int64), np.array([c0 + c2, c1 + c2], np.int64), c0 - c1 - c2, c0 - c1 + c2,
                    np.array([0, c0, c1, c2], np.int64))
        W = (N + 63) // 64
        side_m = np.zeros((k, W), np.uint64)
        dec_m = np.zeros((k, W), np.uint64)
        reg = np.empty((k, 2), np.int64)
        L = max(1, max(p[4].size for p in per))
        leaf = np.zeros((k, L), np.int8)
        spec_ptr = np.zeros(k + 1, np.int64)
        exc_ptr = np.zeros(k + 1, np.int64)
        specs, codes, excs, labs = [], [], [], []
        for i, (Nn, slab_pos, sideb, inv, cls, lab, infam, kind) in enumerate(per):
            if kind == 5:
                sd, dc, spec, code, exc = _expand_sparse(Nn, slab_pos, np.zeros(0, np.int64), sideb, inv,
                                                         np.arange(cls.size, dtype=np.int64), _REV)
            else:
                sd, dc, spec, code, exc = _expand_sparse(Nn, np.zeros(0, np.int64), slab_pos, sideb,
                                                         np.zeros(0, np.int64), np.zeros(0, np.int64), _REV)
            side_m[i] = sd
            dec_m[i] = dc
            reg[i] = (1, 2) if infam else (3, 3)
            leaf[i, :cls.size] = cls
            spec_ptr[i + 1] = spec_ptr[i] + spec.size
            exc_ptr[i + 1] = exc_ptr[i] + exc.size
            specs.append(spec)
            codes.append(code)
            excs.append(exc)
            labs.append(lab)
        spec_idx = np.concatenate(specs + [np.zeros(1, np.int64)])
        spec_code = np.concatenate(codes + [np.zeros(1, np.int32)]).astype(np.int32)
        exc_idx = np.concatenate(excs + [np.zeros(1, np.int64)])
        exc_lab = np.concatenate(labs + [np.zeros(1, np.bool_)])
        return _query_kernel_sp(side_m, dec_m, N, np.arange(k, dtype=np.int64), reg, leaf, spec_ptr, spec_idx,
                                spec_code, exc_ptr, exc_idx, exc_lab, np.zeros(N, np.bool_))
