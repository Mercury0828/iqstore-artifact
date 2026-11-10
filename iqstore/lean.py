"""Lean end-to-end IQSTORE query path (prereg/M2.md section 6: the timed interval
starts at the file on disk; nothing is retained between runs).

Python: one open + one read (`storage.read_file`), decompression of the (rare) zstd-coded blobs, NumPy trig of the stored family bounds and of
h (kept in NumPy so every float equals the encoder's), exceptions' exact labels. Everything else (container index,
header fields, masks, side bits, band ranks, tree, label sets, counts) runs in two compiled calls on the file bytes.
Same answers as the reference path `e2e.IQFileRef` (tests/test_e2e.py); a label set the float filter cannot settle sends
the query to the reference path, whose fallback is exact rational arithmetic.
"""
from __future__ import annotations

import struct

import numba
import numpy as np

from .e2e import _REV, _expand_sparse, _leaf_label_set, _query_kernel_sp, _zload, IQFileRef
from .family import TWO_PI
from .fastdec import _rect_status
from .frame import Frame
from .storage import DTYPES, read_file

_ISZ = np.zeros(256, np.int64)
for _c, _dt in DTYPES.items():
    _ISZ[_c] = np.dtype(_dt).itemsize


@numba.njit(cache=True)
def _decode_tree3(packed, p0, n_bits, X0, Y0, side, cl, sl, ch, sh, th_lo, th_hi, b_lo, b_hi, generic):
    """`fastdec._decode_tree` without per-level allocations: nodes of all levels live in one buffer (a node count is at
    most 1 + 4 * n_bits); same expressions, same order, same output."""
    cap = 1 + 4 * n_bits
    NI = np.empty(cap, np.int64)
    NJ = np.empty(cap, np.int64)
    SP = np.empty(cap, np.bool_)
    lcap = 1 + 3 * n_bits
    ld = np.empty(lcap, np.int64)
    li = np.empty(lcap, np.int64)
    lj = np.empty(lcap, np.int64)
    ls = np.empty(lcap, np.int8)
    nl = 0
    NI[0] = 0
    NJ[0] = 0
    a = 0      # level start
    b = 1      # level end
    d = 0
    pos = 0
    while b > a:
        nsplit = 0
        c = side / (2.0 ** d)
        for k in range(a, b):
            st = _rect_status(X0 + NI[k] * c, Y0 + NJ[k] * c, X0 + (NI[k] + 1) * c, Y0 + (NJ[k] + 1) * c, cl, sl, ch, sh,
                              th_lo, th_hi, b_lo, b_hi)
            if st >= 0 and not generic:
                bit = 0
            else:
                if pos >= n_bits:
                    return ld[:0], li[:0], lj[:0], ls[:0], -1
                bit = (packed[p0 + (pos >> 3)] >> (7 - (pos & 7))) & 1
                pos += 1
            if bit:
                SP[k] = True
                nsplit += 1
            else:
                SP[k] = False
                if nl >= lcap:
                    return ld[:0], li[:0], lj[:0], ls[:0], -1
                ld[nl] = d
                li[nl] = NI[k]
                lj[nl] = NJ[k]
                ls[nl] = st
                nl += 1
        if b + 4 * nsplit > cap:
            return ld[:0], li[:0], lj[:0], ls[:0], -1
        t = b
        e0 = a
        while e0 < b:
            e = e0
            while e < b and NI[e] == NI[e0]:
                e += 1
            for di in range(2):
                for k in range(e0, e):
                    if SP[k]:
                        for dj in range(2):
                            NI[t] = 2 * NI[k] + di
                            NJ[t] = 2 * NJ[k] + dj
                            t += 1
            e0 = e
        a = b
        b = t
        d += 1
    return ld[:nl], li[:nl], lj[:nl], ls[:nl], pos


@numba.njit(cache=True)
def _u32(buf, p):
    return buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16) | (buf[p + 3] << 24)


@numba.njit(cache=True)
def _i64le(buf, p):
    v = 0
    for t in range(8):
        v |= np.int64(buf[p + t]) << (8 * t)
    return v


# loc[i, slot] for slots 0 tree, 1 band mask, 2 exception mask, 3 side bits, 4 band ranks, 5 exceptions raw:
#   [src, start, end, itemsize]; src 0 = file bytes, 1 = aux bytes, -1 = needs decompression of file bytes
#   [start, end) with zload semantics; slot 1/2 kind in col 3 (0 bitmap, else itemsize of the gap list), slot 4 itemsize.
# hdri[i] = (N, n_bits, n_exc, flag, dcode); hdrf[i] = the 11 header doubles.
@numba.njit(cache=True)
def _parse(buf, pos, m, S, isz):
    k = S.size
    loc = np.zeros((k, 6, 4), np.int64)
    hdrf = np.empty((k, 11), np.float64)
    hdri = np.zeros((k, 5), np.int64)
    slot = np.full(m, -1, np.int64)
    last = -1
    for i in range(k):
        slot[S[i]] = i
        if S[i] > last:
            last = S[i]
    tmp = np.empty(88, np.uint8)
    off = np.zeros(16, np.int64)
    ln = np.zeros(16, np.int64)
    for g in range(last + 1):
        L = _u32(buf, pos)
        n = buf[pos + 4] | (buf[pos + 5] << 8)
        pos += 4 + L
        s = slot[g]
        for b in range(n):
            L = _u32(buf, pos)
            if s >= 0 and b < 16:
                off[b] = pos + 4
                ln[b] = L
            pos += 4 + L
        if s < 0:
            continue
        o = off[0]
        for t in range(88):
            tmp[t] = buf[o + t]
        hdrf[s] = tmp.view(np.float64)
        N = _i64le(buf, o + 88)
        n_bits = _i64le(buf, o + 96)
        n_exc = _i64le(buf, o + 104)
        flag = buf[o + 112]
        dcode = buf[o + 113]
        hdri[s, 0] = N
        hdri[s, 1] = n_bits
        hdri[s, 2] = n_exc
        hdri[s, 3] = flag
        hdri[s, 4] = dcode
        # tree
        o = off[1]
        e = o + ln[1]
        loc[s, 0, 0] = 0 if buf[o] == 0 else -1
        loc[s, 0, 1] = o + 1 if buf[o] == 0 else o
        loc[s, 0, 2] = e
        if (flag & 3) == 1:  # single symbol column (slot 1: symbols of `dcode` bytes)
            o = off[2]
            e = o + ln[2]
            loc[s, 1, 0] = 0 if buf[o] == 0 else -1
            loc[s, 1, 1] = o + 1 if buf[o] == 0 else o
            loc[s, 1, 2] = e
            loc[s, 1, 3] = isz[dcode]
            if n_exc:
                o = off[3]
                loc[s, 5, 0] = -1
                loc[s, 5, 1] = o
                loc[s, 5, 2] = o + ln[3]
            continue
        # masks
        for sl, b in ((1, 2), (2, 3)):
            if sl == 2 and n_exc == 0:
                continue
            o = off[b]
            e = o + ln[b]
            if buf[o] == 66:  # 'B' + zload
                loc[s, sl, 3] = 0
                tag = buf[o + 1]
                loc[s, sl, 0] = 0 if tag == 0 else -1
                loc[s, sl, 1] = o + 2 if tag == 0 else o + 1
            else:  # 'I' + dtype code + zload
                loc[s, sl, 3] = isz[buf[o + 1]]
                tag = buf[o + 2]
                loc[s, sl, 0] = 0 if tag == 0 else -1
                loc[s, sl, 1] = o + 3 if tag == 0 else o + 2
            loc[s, sl, 2] = e
        j = 4 if n_exc else 3
        for sl, b in ((3, j), (4, j + 1)):
            o = off[b]
            e = o + ln[b]
            loc[s, sl, 0] = 0 if buf[o] == 0 else -1
            loc[s, sl, 1] = o + 1 if buf[o] == 0 else o
            loc[s, sl, 2] = e
        loc[s, 4, 3] = isz[dcode]
        if n_exc:
            o = off[j + 2]
            loc[s, 5, 0] = -1
            loc[s, 5, 1] = o
            loc[s, 5, 2] = o + ln[j + 2]
    return loc, hdrf, hdri


@numba.njit(cache=True)
def _uint_at(src, p, isz):
    if isz == 1:
        return np.int64(src[p])
    if isz == 2:
        return np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
    v = 0
    for t in range(isz):
        v |= np.int64(src[p + t]) << (8 * t)
    return v


@numba.njit(cache=True)
def _true_pos_b(src, a, e, kind, n):
    """Sorted True positions of a stored mask: MSB-first bitmap (kind 0) or gap list of `kind`-byte integers."""
    if kind == 0:
        cnt = 0
        for t in range(a, e):
            v = src[t]
            while v:
                v &= v - 1
                cnt += 1
        out = np.empty(cnt, np.int64)
        c = 0
        for t in range(a, e):
            by = src[t]
            if by == 0:
                continue
            for b in range(8):
                if (by >> (7 - b)) & 1:
                    s = (t - a) * 8 + b
                    if s < n:
                        out[c] = s
                        c += 1
        return out[:c]
    ng = (e - a) // kind
    out = np.empty(ng, np.int64)
    p = -1
    for t in range(ng):
        p += _uint_at(src, a + t * kind, kind) + 1
        out[t] = p
    return out


@numba.njit(cache=True)
def _count_mask(src, a, e, kind):
    if kind == 0:
        c = 0
        for t in range(a, e):
            v = src[t]
            while v:
                v &= v - 1
                c += 1
        return c
    return (e - a) // kind


@numba.njit(cache=True)
def _cols(buf, aux, loc, hdri, i, N):
    """(band_pos, exc_nb, side bytes, band ranks) of touched position i, from either column layout."""
    if (hdri[i, 3] & 3) == 1:
        src = buf if loc[i, 1, 0] == 0 else aux
        a = loc[i, 1, 1]
        isz = loc[i, 1, 3]
        nb = 0
        ne = 0
        nd = 0
        for s in range(N):
            v = _uint_at(src, a + s * isz, isz)
            if v >= 3:
                nb += 1
            elif v == 2:
                ne += 1
            else:
                nd += 1
        band_pos = np.empty(nb, np.int64)
        bids = np.empty(nb, np.int64)
        exc_nb = np.empty(ne, np.int64)
        sideb = np.zeros((nd + 7) // 8, np.uint8)
        tb = 0
        te = 0
        td = 0
        tn = 0
        for s in range(N):
            v = _uint_at(src, a + s * isz, isz)
            if v >= 3:
                band_pos[tb] = s
                bids[tb] = v - 3
                tb += 1
            else:
                if v == 2:
                    exc_nb[te] = tn
                    te += 1
                else:
                    if v == 1:
                        sideb[td >> 3] |= np.uint8(1 << (7 - (td & 7)))
                    td += 1
                tn += 1
        return band_pos, exc_nb, sideb, bids
    src = buf if loc[i, 1, 0] == 0 else aux
    band_pos = _true_pos_b(src, loc[i, 1, 1], loc[i, 1, 2], loc[i, 1, 3], N)
    n_nb = N - band_pos.size
    if hdri[i, 2] > 0:
        src = buf if loc[i, 2, 0] == 0 else aux
        exc_nb = _true_pos_b(src, loc[i, 2, 1], loc[i, 2, 2], loc[i, 2, 3], n_nb)
    else:
        exc_nb = np.zeros(0, np.int64)
    src = buf if loc[i, 3, 0] == 0 else aux
    sideb = src[loc[i, 3, 1]:loc[i, 3, 2]].copy()
    rsrc = buf if loc[i, 4, 0] == 0 else aux
    isz = loc[i, 4, 3]
    nr = (loc[i, 4, 2] - loc[i, 4, 1]) // isz
    bids = np.empty(nr, np.int64)
    for t in range(nr):
        bids[t] = _uint_at(rsrc, loc[i, 4, 1] + t * isz, isz)
    return band_pos, exc_nb, sideb, bids


@numba.njit(cache=True)
def _expand_sym(src, a, isz, N, band_leaves):
    """Single pass over a stored symbol column (0/1 decided side, 2 exception, 3 + r band rank r): the outputs of
    `e2e._expand_sparse` (side / decided words, special shots with cell codes, exception shots) directly."""
    W = (N + 63) // 64
    side = np.zeros(W, np.uint64)
    dec = np.zeros(W, np.uint64)
    cap = 64
    spec = np.empty(cap, np.int64)
    code = np.empty(cap, np.int32)
    exc = np.empty(cap, np.int64)
    ns = 0
    ne = 0
    for w in range(W):
        lo = w * 64
        hi = lo + 64
        if hi > N:
            hi = N
        sw = np.uint64(0)
        dw = np.uint64(0)
        for s in range(lo, hi):
            if isz == 1:
                v = np.int64(src[a + s])
            else:
                v = _uint_at(src, a + s * isz, isz)
            bit = np.uint64(1) << np.uint64(s - lo)
            if v <= 1:
                dw |= bit
                if v == 1:
                    sw |= bit
            else:
                if ns == cap:
                    cap *= 2
                    s2 = np.empty(cap, np.int64)
                    c2 = np.empty(cap, np.int32)
                    e2 = np.empty(cap, np.int64)
                    s2[:ns] = spec[:ns]
                    c2[:ns] = code[:ns]
                    e2[:ne] = exc[:ne]
                    spec, code, exc = s2, c2, e2
                spec[ns] = s
                if v == 2:
                    code[ns] = 2
                    exc[ne] = s
                    ne += 1
                else:
                    if v - 3 >= band_leaves.size:
                        return side, dec, spec[:0], code[:0], exc[:0], False
                    code[ns] = 3 + band_leaves[v - 3]
                ns += 1
        side[w] = sw
        dec[w] = dw
    return side, dec, spec[:ns], code[:ns], exc[:ne], True


@numba.njit(cache=True)
def _in_family(theta, beta, th_lo, th_hi, b_lo, b_hi):
    kk = np.ceil((th_lo - theta) / TWO_PI)
    t2 = theta + kk * TWO_PI
    return t2 <= th_hi and b_lo <= beta and beta <= b_hi


@numba.njit(cache=True)
def _query_buf(buf, aux, loc, hdrf, hdri, trig, hb, exl, exl_ptr, rev):
    """trig[i] = (cos th_lo, sin th_lo, cos th_hi, sin th_hi, p, q) from NumPy; hb[i] = (theta, beta) of h_i.
    Returns (lo, hi, par_lo, par_hi, status): 0 ok, 1 exact arithmetic needed (general path), 2 bad stream."""
    k = loc.shape[0]
    N = hdri[0, 0]
    P = 1 << k
    fq = np.empty(14, np.float64)
    if k == 1:
        th_lo, th_hi, b_lo, b_hi = hdrf[0, 7], hdrf[0, 8], hdrf[0, 9], hdrf[0, 10]
        if _in_family(hb[0, 0], hb[0, 1], th_lo, th_hi, b_lo, b_hi):
            n_exc = hdri[0, 2]
            sym = (hdri[0, 3] & 3) == 1
            hist = np.zeros(3, np.int64)
            if sym:
                src = buf if loc[0, 1, 0] == 0 else aux
                a = loc[0, 1, 1]
                isz = loc[0, 1, 3]
                if isz == 1:
                    h256 = np.zeros(256, np.int64)
                    for s_ in range(N):
                        h256[src[a + s_]] += 1
                    vmax = 2
                    for v in range(255, 2, -1):
                        if h256[v] > 0:
                            vmax = v
                            break
                    hist = h256[:vmax + 1].copy()
                else:
                    vmax = 2
                    for s_ in range(N):
                        v = _uint_at(src, a + s_ * isz, isz)
                        if v > vmax:
                            vmax = v
                    hist = np.zeros(vmax + 1, np.int64)
                    for s_ in range(N):
                        hist[_uint_at(src, a + s_ * isz, isz)] += 1
                n_band = N - hist[0] - hist[1] - hist[2]
                c0 = hist[0]
                c1 = hist[1]
            else:
                src = buf if loc[0, 1, 0] == 0 else aux
                n_band = _count_mask(src, loc[0, 1, 1], loc[0, 1, 2], loc[0, 1, 3])
                src = buf if loc[0, 3, 0] == 0 else aux
                n1 = _count_mask(src, loc[0, 3, 1], loc[0, 3, 2], 0)
                c0 = N - n_band - n_exc - n1
                c1 = n1
            c2 = 0
            for t in range(exl_ptr[0], exl_ptr[1]):
                if exl[t]:
                    c1 += 1
                else:
                    c0 += 1
            if n_band > 0:
                n_bits = hdri[0, 1]
                tsrc = buf if loc[0, 0, 0] == 0 else aux
                D, I, J, St, pos = _decode_tree3(tsrc, loc[0, 0, 1], n_bits, hdrf[0, 0], hdrf[0, 1], hdrf[0, 2],
                                                 trig[0, 0], trig[0, 1], trig[0, 2], trig[0, 3], th_lo, th_hi, b_lo,
                                                 b_hi, (hdri[0, 3] & 4) != 0)
                if pos != n_bits:
                    return np.zeros(2, np.int64), np.zeros(2, np.int64), 0, 0, np.zeros(1, np.int64), 2
                nbl = 0
                for a in range(I.size):
                    if St[a] == -1:
                        nbl += 1
                w = np.zeros(nbl, np.int64)
                if sym:
                    if hist.size - 3 > nbl:
                        return np.zeros(2, np.int64), np.zeros(2, np.int64), 0, 0, np.zeros(1, np.int64), 2
                    for r in range(3, hist.size):
                        w[r - 3] = hist[r]
                else:
                    rsrc = buf if loc[0, 4, 0] == 0 else aux
                    isz = loc[0, 4, 3]
                    for p in range(loc[0, 4, 1], loc[0, 4, 2], isz):
                        r = _uint_at(rsrc, p, isz)
                        if r >= nbl:
                            return np.zeros(2, np.int64), np.zeros(2, np.int64), 0, 0, np.zeros(1, np.int64), 2
                        w[r] += 1
                r = 0
                for a in range(I.size):
                    if St[a] == -1:
                        if w[r] > 0:
                            v = _leaf_label_set(hdrf[0, 0], hdrf[0, 1], hdrf[0, 2], D[a], I[a], J[a], trig[0, 4],
                                                trig[0, 5], hb[0, 1])
                            if v < 0:
                                return np.zeros(2, np.int64), np.zeros(2, np.int64), 0, 0, np.zeros(1, np.int64), 1
                            if v == 1:
                                c0 += w[r]
                            elif v == 2:
                                c1 += w[r]
                            else:
                                c2 += w[r]
                        r += 1
            lo = np.zeros(2, np.int64)
            hi = np.zeros(2, np.int64)
            lo[0] = c0
            lo[1] = c1
            hi[0] = c0 + c2
            hi[1] = c1 + c2
            tup = np.zeros(4, np.int64)
            tup[1] = c0
            tup[2] = c1
            tup[3] = c2
            return lo, hi, c0 - c1 - c2, c0 - c1 + c2, tup, 0
    W = (N + 63) // 64
    side_m = np.zeros((k, W), np.uint64)
    dec_m = np.zeros((k, W), np.uint64)
    reg = np.empty((k, 2), np.int64)
    spec_l = []
    code_l = []
    exc_l = []
    leaf_l = []
    L = 1
    for i in range(k):
        th_lo, th_hi, b_lo, b_hi = hdrf[i, 7], hdrf[i, 8], hdrf[i, 9], hdrf[i, 10]
        n_bits = hdri[i, 1]
        tsrc = buf if loc[i, 0, 0] == 0 else aux
        D, I, J, St, pos = _decode_tree3(tsrc, loc[i, 0, 1], n_bits, hdrf[i, 0], hdrf[i, 1], hdrf[i, 2], trig[i, 0],
                                         trig[i, 1], trig[i, 2], trig[i, 3], th_lo, th_hi, b_lo, b_hi,
                                         (hdri[i, 3] & 4) != 0)
        if pos != n_bits:
            return np.zeros(P, np.int64), np.zeros(P, np.int64), 0, 0, np.zeros(1, np.int64), 2
        nl = I.size
        if nl > L:
            L = nl
        nband = 0
        for a in range(nl):
            if St[a] == -1:
                nband += 1
        band_leaves = np.empty(nband, np.int64)
        t = 0
        for a in range(nl):
            if St[a] == -1:
                band_leaves[t] = a
                t += 1
        if (hdri[i, 3] & 3) == 1:
            src = buf if loc[i, 1, 0] == 0 else aux
            sd, dc, spec, code, exc, ok = _expand_sym(src, loc[i, 1, 1], loc[i, 1, 3], N, band_leaves)
            if not ok:
                return np.zeros(P, np.int64), np.zeros(P, np.int64), 0, 0, np.zeros(1, np.int64), 2
        else:
            band_pos, exc_nb, sideb, bids = _cols(buf, aux, loc, hdri, i, N)
            sd, dc, spec, code, exc = _expand_sparse(N, band_pos, exc_nb, sideb, bids, band_leaves, rev)
        side_m[i] = sd
        dec_m[i] = dc
        leaf = np.zeros(nl, np.int8)
        infam = _in_family(hb[i, 0], hb[i, 1], th_lo, th_hi, b_lo, b_hi)
        r0 = 0
        r1 = 0
        for a in range(nl):
            if St[a] == -1:
                v = _leaf_label_set(hdrf[i, 0], hdrf[i, 1], hdrf[i, 2], D[a], I[a], J[a], trig[i, 4], trig[i, 5],
                                    hb[i, 1])
                if v < 0:
                    return np.zeros(P, np.int64), np.zeros(P, np.int64), 0, 0, np.zeros(1, np.int64), 1
                leaf[a] = v
            elif not infam:
                v = _leaf_label_set(hdrf[i, 0], hdrf[i, 1], hdrf[i, 2], D[a], I[a], J[a], trig[i, 4], trig[i, 5],
                                    hb[i, 1])
                if v < 0:
                    return np.zeros(P, np.int64), np.zeros(P, np.int64), 0, 0, np.zeros(1, np.int64), 1
                if St[a] == 0:
                    r0 |= v
                else:
                    r1 |= v
        if infam:
            reg[i, 0] = 1
            reg[i, 1] = 2
        else:
            reg[i, 0] = r0 if r0 != 0 else 1
            reg[i, 1] = r1 if r1 != 0 else 2
        spec_l.append(spec)
        code_l.append(code)
        exc_l.append(exc)
        leaf_l.append(leaf)
    leaf_m = np.zeros((k, L), np.int8)
    spec_ptr = np.zeros(k + 1, np.int64)
    exc_ptr = np.zeros(k + 1, np.int64)
    for i in range(k):
        leaf_m[i, :leaf_l[i].size] = leaf_l[i]
        spec_ptr[i + 1] = spec_ptr[i] + spec_l[i].size
        exc_ptr[i + 1] = exc_ptr[i] + exc_l[i].size
    spec_idx = np.zeros(max(spec_ptr[k], 1), np.int64)
    spec_code = np.zeros(max(spec_ptr[k], 1), np.int32)
    exc_idx = np.zeros(max(exc_ptr[k], 1), np.int64)
    exc_lab = np.zeros(max(exc_ptr[k], 1), np.bool_)
    for i in range(k):
        spec_idx[spec_ptr[i]:spec_ptr[i + 1]] = spec_l[i]
        spec_code[spec_ptr[i]:spec_ptr[i + 1]] = code_l[i]
        exc_idx[exc_ptr[i]:exc_ptr[i + 1]] = exc_l[i]
        exc_lab[exc_ptr[i]:exc_ptr[i + 1]] = exl[exl_ptr[i]:exl_ptr[i + 1]]
    Sx = np.arange(k)
    lo, hi, plo, phi, tup = _query_kernel_sp(side_m, dec_m, N, Sx, reg, leaf_m, spec_ptr, spec_idx, spec_code,
                                             exc_ptr, exc_idx, exc_lab, np.zeros(N, np.bool_))
    return lo, hi, plo, phi, tup, 0


_HEAD = struct.Struct("<qqI")
_EXL0 = np.zeros(1, np.bool_)
_AUX0 = np.zeros(1, np.uint8)


class IQFile:
    """IQSTORE engine over a persisted container (one file per dataset; groups = positions in order)."""

    def __init__(self, path):
        self.path = path
        self._slow = IQFileRef(path)

    def query(self, S, hs):
        data = read_file(self.path)
        buf = np.frombuffer(data, dtype=np.uint8)
        ln = data[0]
        N, m, nb = _HEAD.unpack_from(data, 1 + ln)
        k = len(S)
        loc, hdrf, hdri = _parse(buf, 1 + ln + 20, m, np.array(S, dtype=np.int64), _ISZ)
        # one vectorized cos / sin call over all 3k angles (bit-identical to the scalar calls;
        # tests/test_e2e.py), one check for zstd slots and exceptions instead of per-position Python work.
        hb = np.array([(h.theta, h.beta) for h in hs], dtype=np.float64)
        ang = np.empty((k, 3))
        ang[:, 0] = hdrf[:, 7]
        ang[:, 1] = hdrf[:, 8]
        ang[:, 2] = hb[:, 0]
        c = np.cos(ang)
        s = np.sin(ang)
        trig = np.empty((k, 6))
        trig[:, 0::2] = c
        trig[:, 1::2] = s
        aux_parts = None
        exls = None
        need = (loc[:, :5, 0] == -1).any(axis=1) | (hdri[:, 2] != 0)
        for i in np.flatnonzero(need).tolist():
            li = loc[i]
            h = hs[i]
            f = hdrf[i]
            if (li[:5, 0] == -1).any():
                if aux_parts is None:
                    aux_parts = []
                    aux_n = 0
                for sl in range(5):
                    if li[sl, 0] == -1:
                        pay = _zload(memoryview(data)[li[sl, 1]:li[sl, 2]])
                        li[sl, 0] = 1
                        li[sl, 1] = aux_n
                        aux_parts.append(pay)
                        aux_n += len(pay)
                        li[sl, 2] = aux_n
            if hdri[i, 2]:
                if exls is None:
                    exls = [_EXL0[:0]] * k
                blob = memoryview(data)[li[5, 1]:li[5, 2]]
                fr = Frame.from_array(f[3:7])
                ex, ey = fr.apply(np.frombuffer(_zload(blob[1:]), dtype=DTYPES[blob[0]]))
                exls[i] = np.ascontiguousarray(h.label(ex, ey), dtype=np.bool_)
        aux = np.frombuffer(b"".join(aux_parts) + b"\0", dtype=np.uint8) if aux_parts else _AUX0
        if exls is None:
            exl = _EXL0
            exl_ptr = np.zeros(k + 1, np.int64)
        else:
            exl = np.concatenate(exls + [_EXL0])
            exl_ptr = np.zeros(k + 1, np.int64)
            exl_ptr[1:] = np.cumsum([e.size for e in exls])
        lo, hi, plo, phi, tup, st = _query_buf(buf, aux, loc, hdrf, hdri, trig, hb, exl, exl_ptr, _REV)
        if st == 0:
            return lo, hi, plo, phi, tup
        if st == 2:
            raise ValueError("tree bit count mismatch")
        return self._slow.query(S, hs)
