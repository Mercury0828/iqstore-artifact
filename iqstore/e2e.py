"""End-to-end IQSTORE query from the persisted container (prereg/M2.md section 6: the timed interval starts at the file
on disk and includes decode, index construction, geometry, label sets and aggregation; nothing is retained between
runs). Only the touched positions are decoded; the tree is decoded by the compiled
kernel of `fastdec`; the per-shot columns are expanded by one compiled pass straight into the query kernel's inputs
(side / decided bit words, cell codes, special-shot list), without the intermediate symbol arrays of
`storage.decode_iqstore_position`. Results equal `FastDataset.query` on the decoded representation (tested).
"""
from __future__ import annotations

import os
import struct

import numba
import numpy as np
import zstandard as zstd

from .family import Discriminator, Family, unit
from .fastdec import _decode_tree, decode_tree
from .fastq import _popc
from .frame import Frame
from .rep import LS_ONLY0, LS_ONLY1, Partition, rect_label_sets
from .storage import DTYPES, _IQ_HDR, _unshuffle, read_file as _read

_HDR = struct.Struct(_IQ_HDR)
_U32 = struct.Struct("<I")
_DZ = zstd.ZstdDecompressor()


def _zload(mv) -> bytes:
    tag = mv[0]
    if tag == 0:
        return bytes(mv[1:])
    if tag == 1:
        return _DZ.decompress(mv[1:])
    return _unshuffle(_DZ.decompress(mv[2:]), mv[1])


def _mask(mv, n: int):
    """(kind, payload): kind 0 = MSB-first packed bitmap bytes, kind 1 = gap list (int64)."""
    if mv[0] == 66:  # b'B'
        return 0, np.frombuffer(_zload(mv[1:]), dtype=np.uint8)
    return 1, np.frombuffer(_zload(mv[2:]), dtype=DTYPES[mv[1]]).astype(np.int64)


_REV = np.array([int(f"{v:08b}"[::-1], 2) for v in range(256)], dtype=np.uint8)  # bit reversal of a byte


@numba.njit(cache=True)
def _true_positions(kind, payload, n):
    """Sorted positions of the True entries of a stored mask (MSB-first bitmap, or gap list)."""
    if kind == 0:
        cnt = 0
        for by in payload:
            v = by
            while v:
                v &= v - 1
                cnt += 1
        out = np.empty(cnt, np.int64)
        t = 0
        for i in range(payload.size):
            by = payload[i]
            if by == 0:
                continue
            for b in range(8):
                if (by >> (7 - b)) & 1:
                    s = i * 8 + b
                    if s < n:
                        out[t] = s
                        t += 1
        return out[:t]
    out = np.empty(payload.size, np.int64)
    p = -1
    for i in range(payload.size):
        p += payload[i] + 1
        out[i] = p
    return out


@numba.njit(cache=True)
def _expand_sparse(N, band_pos, exc_nb, side_bytes, band_ids, band_leaves, rev):
    """Packed (little-endian) side / decided words and the special shots (band or exception) with their cell codes.
    band_pos: sorted band shots; exc_nb: sorted exception indices among non-band shots; side_bytes: MSB-first side bits of
    the decided shots in shot order; band_ids: band rank per band shot (shot order); band_leaves: leaf index per rank."""
    W = (N + 63) // 64
    nb = band_pos.size
    ne = exc_nb.size
    # exception shot ids: non-band index t -> shot s with s - #{band shots < s} = t
    exc = np.empty(ne, np.int64)
    bi = 0
    for e in range(ne):
        t = exc_nb[e]
        s = t + bi
        while bi < nb and band_pos[bi] <= s:
            bi += 1
            s = t + bi
        exc[e] = s
    # special shots in shot order, with codes
    spec = np.empty(nb + ne, np.int64)
    code = np.empty(nb + ne, np.int32)
    i = 0
    e = 0
    t = 0
    while i < nb or e < ne:
        if e >= ne or (i < nb and band_pos[i] < exc[e]):
            spec[t] = band_pos[i]
            code[t] = 3 + band_leaves[band_ids[i]]
            i += 1
        else:
            spec[t] = exc[e]
            code[t] = 2
            e += 1
        t += 1
    spm = np.zeros(W, np.uint64)
    for s in spec:
        spm[s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    # side stream as LSB-first 64-bit words
    nbytes = side_bytes.size
    A = np.zeros((nbytes + 7) // 8 + 2, np.uint64)
    for k in range(nbytes):
        A[k >> 3] |= np.uint64(rev[side_bytes[k]]) << np.uint64(8 * (k & 7))
    side = np.zeros(W, np.uint64)
    dec = np.zeros(W, np.uint64)
    off = 0
    for w in range(W):
        nvalid = N - 64 * w
        if nvalid > 64:
            nvalid = 64
        valid = np.uint64(0xFFFFFFFFFFFFFFFF) if nvalid == 64 else (np.uint64(1) << np.uint64(nvalid)) - np.uint64(1)
        d = valid & ~spm[w]
        dec[w] = d
        cnt = 0
        x = d
        while x:
            x &= x - np.uint64(1)
            cnt += 1
        if cnt == 0:
            continue
        q = off >> 6
        r = off & 63
        v = A[q] >> np.uint64(r)
        if r and cnt > 64 - r:
            v |= A[q + 1] << np.uint64(64 - r)
        if cnt < 64:
            v &= (np.uint64(1) << np.uint64(cnt)) - np.uint64(1)
        off += cnt
        if d == valid and nvalid == 64:
            side[w] = v
        else:
            out = np.uint64(0)
            for b in range(nvalid):
                bit = np.uint64(1) << np.uint64(b)
                if d & bit:
                    if v & np.uint64(1):
                        out |= bit
                    v >>= np.uint64(1)
            side[w] = out
    return side, dec, spec, code, exc


class _Pos:
    __slots__ = ("frame", "fam", "part", "side", "dec", "spec", "code", "exc_index", "exc_x", "exc_y")


def _decode(blobs, N_expected: int) -> _Pos:
    h = _HDR.unpack(blobs[0])
    root, fam = h[0:3], Family(*h[7:11])
    N, n_bits, n_exc, flag, dcode = h[11], h[12], h[13], h[14], h[15]
    generic = flag >= 4
    flag -= 4 if generic else 0
    D, I, J, S = decode_tree(_zload(blobs[1]), n_bits, root, fam, generic)
    band_leaves = np.flatnonzero(S == -1)
    if flag == 1:  # single symbol column (chosen only when smaller)
        sym = np.frombuffer(_zload(blobs[2]), dtype=DTYPES[dcode]).astype(np.int64)
        band = sym >= 3
        nbm = ~band
        band_pos = np.flatnonzero(band)
        exc_nb = np.flatnonzero(sym[nbm] == 2)
        side_bits = np.packbits(sym[nbm][sym[nbm] != 2] == 1)
        band_ids = sym[band] - 3
        rest = blobs[3:]
    else:
        k, pay = _mask(blobs[2], N)
        band_pos = _true_positions(k, pay, N)
        j = 3
        if n_exc:
            k, pay = _mask(blobs[3], N - band_pos.size)
            exc_nb = _true_positions(k, pay, N - band_pos.size)
            j = 4
        else:
            exc_nb = np.zeros(0, np.int64)
        side_bits = np.frombuffer(_zload(blobs[j]), dtype=np.uint8)
        band_ids = np.frombuffer(_zload(blobs[j + 1]), dtype=DTYPES[dcode]).astype(np.int64)
        rest = blobs[j + 2:]
    p = _Pos()
    p.side, p.dec, p.spec, p.code, p.exc_index = _expand_sparse(N, band_pos, exc_nb, side_bits, band_ids, band_leaves,
                                                                _REV)
    p.fam = fam
    p.part = Partition(tuple(root), I, J, D, S)
    if n_exc:
        fr = Frame.from_array(np.array(h[3:7]))
        raw = np.frombuffer(_zload(rest[0][1:]), dtype=DTYPES[rest[0][0]])
        p.exc_x, p.exc_y = fr.apply(raw)
    else:
        p.exc_x = p.exc_y = np.zeros(0)
    return p


_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)


@numba.njit(cache=True)
def _sign_filter(x, y, p, q, b):
    """`family.exact_sign` float filter: returns the exact sign, or 2 when the float value is inside its error bound."""
    px = p * x
    qy = q * y
    d = (px + qy) - b
    err = 2.0 * _EPS * (abs(px) + abs(qy) + abs(b)) + _TINY
    if not (abs(d) > err):
        return 2
    return 1 if d > 0 else (-1 if d < 0 else 0)


@numba.njit(cache=True)
def _band_label_sets(X0, Y0, side, D, I, J, S, p, q, beta, out, unsure):
    """`rep.rect_label_sets` for the band leaves (status -1), same corner choice; leaves whose comparison the float
    filter cannot settle are listed in `unsure` (resolved exactly by the caller). Returns the number of unsure leaves."""
    nu = 0
    for k in range(I.size):
        if S[k] != -1:
            continue
        c = side / (2.0 ** D[k])
        x0 = X0 + I[k] * c
        x1 = X0 + (I[k] + 1) * c
        y0 = Y0 + J[k] * c
        y1 = Y0 + (J[k] + 1) * c
        xu = x1 if p > 0 else x0
        yu = y1 if q > 0 else y0
        xl = x0 if p >= 0 else x1
        yl = y0 if q >= 0 else y1
        su = _sign_filter(xu, yu, p, q, beta)
        sl = _sign_filter(xl, yl, p, q, beta)
        if su == 2 or sl == 2:
            unsure[nu] = k
            nu += 1
            continue
        v = 0
        if sl < 0 or (sl == 0 and p >= 0 and q >= 0):
            v |= 1
        if su > 0:
            v |= 2
        out[k] = v
    return nu


def _tables(p: _Pos, h: Discriminator):
    part = p.part
    if p.fam.contains(h):
        reg = (LS_ONLY0, LS_ONLY1)
    else:
        reg = []
        for s in (0, 1):
            sel = np.flatnonzero(part.status == s)
            ls = rect_label_sets(part.x0[sel], part.y0[sel], part.x1[sel], part.y1[sel], h.theta, h.beta)
            reg.append(int(np.bitwise_or.reduce(ls)) if sel.size else LS_ONLY0 + s)
    pp, qq = unit(h.theta)
    leaf_ls = np.zeros(part.n_leaves, dtype=np.int8)
    unsure = np.empty(part.n_leaves, dtype=np.int64)
    nu = _band_label_sets(float(part.root[0]), float(part.root[1]), float(part.root[2]), part.depth, part.ix, part.iy,
                          part.status, pp, qq, float(h.beta), leaf_ls, unsure)
    if nu:
        u = unsure[:nu]
        sub = Partition(part.root, part.ix[u], part.iy[u], part.depth[u], part.status[u])
        leaf_ls[u] = rect_label_sets(sub.x0, sub.y0, sub.x1, sub.y1, h.theta, h.beta)
    return tuple(reg), leaf_ls


@numba.njit(cache=True)
def _blob_index(buf, pos, m, want):
    """Offsets / lengths of the blobs of the wanted positions (want: (m,) int64, -1 = skip; else output slot).
    Container layout: per position a 2-byte group header blob (number of blobs) then that many blobs, each blob preceded
    by its uint32 little-endian length."""
    k = 0
    for j in range(m):
        if want[j] >= 0:
            k += 1
    off = np.zeros((k, 16), np.int64)
    ln = np.zeros((k, 16), np.int64)
    cnt = np.zeros(k, np.int64)
    for j in range(m):
        L = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16) | (buf[pos + 3] << 24)
        n = buf[pos + 4] | (buf[pos + 5] << 8)
        pos += 4 + L
        slot = want[j]
        if n > 16:
            raise ValueError("more than 16 blobs in a position group")
        for b in range(n):
            L = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16) | (buf[pos + 3] << 24)
            if slot >= 0:
                off[slot, b] = pos + 4
                ln[slot, b] = L
            pos += 4 + L
        if slot >= 0:
            cnt[slot] = n
    return off, ln, cnt


def read_groups(path, S: list[int]):
    """The blob groups of positions S from an IQSTORE container (memoryviews of their bytes)."""
    data = _read(path)
    mv = memoryview(data)
    ln = data[0]
    N, m, nb = struct.unpack_from("<qqI", data, 1 + ln)
    want = np.full(m, -1, np.int64)
    for i, j in enumerate(S):
        want[j] = i
    off, lens, cnt = _blob_index(np.frombuffer(data, dtype=np.uint8), 1 + ln + 20, m, want)
    out = []
    for i in range(len(S)):
        out.append([mv[o:o + L] for o, L in zip(off[i, :cnt[i]].tolist(), lens[i, :cnt[i]].tolist())])
    return N, out


# Same kernel as `fastq._query_kernel`, reading a shot's cell from the bit words and the sparse special-shot list
# instead of a dense per-shot code array.
@numba.njit(cache=True)
def _query_kernel_sp(side, dec, N, Sx, reg, leaf_ls, spec_ptr, spec_idx, spec_code, exc_ptr, exc_idx, exc_lab, mark):
    """All statistics of one pattern-count query in one pass. side/dec: (m, W) uint64; codes: (m, N) int32; Sx: (k,)
    positions; reg: (k, 2) region label sets; leaf_ls: (k, L) leaf label sets; spec_*: CSR of special shots per position
    of the dataset; exc_*: CSR of exception shots (sorted) and their exact labels under h (per query position i);
    mark: (N,) bool scratch (all False on entry and on exit).
    Returns (lo, hi, par_lo, par_hi, tup): pattern-count bounds, parity bounds and, for k <= 6, the number of shots per
    label-set tuple (index sum_i code_i 4^i, code 1 = {0}, 2 = {1}, 3 = {0, 1}; the input of an exact interval of any
    linear statistic, e.g. a mitigated expectation); for k > 6 tup has one zero entry."""
    k = Sx.shape[0]
    P = 1 << k
    lo = np.zeros(P, dtype=np.int64)
    hi = np.zeros(P, dtype=np.int64)
    par_lo = 0
    par_hi = 0
    TK = k <= 6
    tup = np.zeros((1 << (2 * k)) if TK else 1, dtype=np.int64)
    all_single = True
    for i in range(k):
        if reg[i, 0] != 1 or reg[i, 1] != 2:
            all_single = False
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
        if TK:
            for p in range(P):
                if lo[p]:
                    key = 0
                    for i in range(k):
                        key |= (2 if (p >> i) & 1 else 1) << (2 * i)
                    tup[key] += lo[p]
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
            w = s >> 6
            bit = np.uint64(1) << np.uint64(s & 63)
            if dec[j, w] & bit:
                c = 1 if side[j, w] & bit else 0
            else:  # binary search in the sorted special shots of position j
                a2 = spec_ptr[j]
                b2 = spec_ptr[j + 1]
                while a2 < b2:
                    mid = (a2 + b2) // 2
                    if spec_idx[mid] < s:
                        a2 = mid + 1
                    else:
                        b2 = mid
                c = spec_code[a2]
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
        if TK:
            key = 0
            for i in range(k):
                key |= (((can0 >> i) & 1) | (((can1 >> i) & 1) << 1)) << (2 * i)
            tup[key] += 1
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
    return lo, hi, par_lo, par_hi, tup



@numba.njit(cache=True)
def _pos_all(tree, n_bits, fp, generic, N, bk, b_u8, b_i64, has_exc, ek, e_u8, e_i64, side_bytes, band_ids, rev):
    """One compiled call per touched position: tree, columns, and the band leaves' label sets under h.
    fp = (X0, Y0, side, cos lo, sin lo, cos hi, sin hi, th_lo, th_hi, b_lo, b_hi, p, q, beta)."""
    D, I, J, S, pos = _decode_tree(tree, n_bits, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8], fp[9],
                                   fp[10], generic)
    nl = I.size
    nband = 0
    for k in range(nl):
        if S[k] == -1:
            nband += 1
    band_leaves = np.empty(nband, np.int64)
    t = 0
    for k in range(nl):
        if S[k] == -1:
            band_leaves[t] = k
            t += 1
    band_pos = _true_positions(bk, b_u8, N) if bk == 0 else _true_positions(bk, b_i64, N)
    n_nb = N - band_pos.size
    if has_exc:
        exc_nb = _true_positions(ek, e_u8, n_nb) if ek == 0 else _true_positions(ek, e_i64, n_nb)
    else:
        exc_nb = np.zeros(0, np.int64)
    side, dec, spec, code, exc = _expand_sparse(N, band_pos, exc_nb, side_bytes, band_ids, band_leaves, rev)
    leaf_ls = np.zeros(nl, np.int8)
    unsure = np.empty(nl, np.int64)
    nu = _band_label_sets(fp[0], fp[1], fp[2], D, I, J, S, fp[11], fp[12], fp[13], leaf_ls, unsure)
    return D, I, J, S, pos, side, dec, spec, code, exc, leaf_ls, unsure[:nu]


_E_U8 = np.frombuffer(bytes(1), dtype=np.uint8)  # read-only like every frombuffer payload (uniform tuple types)
_E_I64 = np.zeros(1, np.int64)


def _mask2(mv):
    """(kind, uint8 payload, int64 payload) of a stored mask, for `_pos_all`."""
    if mv[0] == 66:  # b'B'
        return 0, np.frombuffer(_zload(mv[1:]), dtype=np.uint8), _E_I64
    return 1, _E_U8, np.frombuffer(_zload(mv[2:]), dtype=DTYPES[mv[1]]).astype(np.int64)


def _decode_query(blobs, h: Discriminator):
    """Decode one touched position and build its tables under h. Returns (side, dec, spec, code, exc_index, exc_labels,
    reg, leaf_ls, n_leaves)."""
    hd = _HDR.unpack(blobs[0])
    th_lo, th_hi, b_lo, b_hi = hd[7:11]
    N, n_bits, n_exc, flag, dcode = hd[11], hd[12], hd[13], hd[14], hd[15]
    generic = flag >= 4
    flag -= 4 if generic else 0
    pp, qq = unit(h.theta)
    fp = np.array((hd[0], hd[1], hd[2], np.cos(th_lo), np.sin(th_lo), np.cos(th_hi), np.sin(th_hi), th_lo, th_hi, b_lo,
                   b_hi, pp, qq, h.beta), dtype=np.float64)
    tree = _zload(blobs[1])
    tree = np.frombuffer(tree, dtype=np.uint8) if len(tree) else _E_U8
    if flag == 1:
        sym = np.frombuffer(_zload(blobs[2]), dtype=DTYPES[dcode]).astype(np.int64)
        band = sym >= 3
        nbs = sym[~band]
        bk, bu, bi = 1, _E_U8, np.diff(np.flatnonzero(band), prepend=-1) - 1
        has_exc = bool((nbs == 2).any())
        ek, eu, ei = 1, _E_U8, np.diff(np.flatnonzero(nbs == 2), prepend=-1) - 1
        side_bits = np.packbits(nbs[nbs != 2] == 1)
        band_ids = sym[band] - 3
        rest = blobs[3:]
    else:
        bk, bu, bi = _mask2(blobs[2])
        j = 3
        has_exc = n_exc > 0
        if has_exc:
            ek, eu, ei = _mask2(blobs[3])
            j = 4
        else:
            ek, eu, ei = 0, _E_U8, _E_I64
        side_bits = np.frombuffer(_zload(blobs[j]), dtype=np.uint8)
        band_ids = np.frombuffer(_zload(blobs[j + 1]), dtype=DTYPES[dcode]).astype(np.int64)
        rest = blobs[j + 2:]
    if side_bits.size == 0:
        side_bits = _E_U8
    D, I, J, S, pos, side, dec, spec, code, exc, leaf_ls, unsure = _pos_all(
        tree, n_bits, fp, generic, N, bk, bu, bi, has_exc, ek, eu, ei, side_bits, band_ids, _REV)
    if pos != n_bits:
        raise ValueError("tree bit count mismatch")
    root = (hd[0], hd[1], hd[2])
    if unsure.size:  # exact resolution of the few comparisons the float filter could not settle
        sub = Partition(root, I[unsure], J[unsure], D[unsure], S[unsure])
        leaf_ls[unsure] = rect_label_sets(sub.x0, sub.y0, sub.x1, sub.y1, h.theta, h.beta)
    fam = Family(th_lo, th_hi, b_lo, b_hi)
    if fam.contains(h):
        reg = (LS_ONLY0, LS_ONLY1)
    else:
        part = Partition(root, I, J, D, S)
        reg = []
        for s in (0, 1):
            sel = np.flatnonzero(S == s)
            ls = rect_label_sets(part.x0[sel], part.y0[sel], part.x1[sel], part.y1[sel], h.theta, h.beta)
            reg.append(int(np.bitwise_or.reduce(ls)) if sel.size else LS_ONLY0 + s)
        reg = tuple(reg)
    if n_exc:
        fr = Frame.from_array(np.array(hd[3:7]))
        ex, ey = fr.apply(np.frombuffer(_zload(rest[0][1:]), dtype=DTYPES[rest[0][0]]))
        exc_lab = h.label(ex, ey)
    else:
        exc_lab = np.zeros(0, np.bool_)
    return side, dec, spec, code, exc, exc_lab, reg, leaf_ls, I.size, N




@numba.njit(cache=True)
def _leaf_label_set(X0, Y0, side, d, i, j, p, q, beta):
    """`rep.rect_label_sets` of one leaf with the float filter; -1 when the filter cannot settle it."""
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
def _query_all(trees, fps, ips, bu8s, bi64s, eu8s, ei64s, sides, bids, exclabs, rev):
    """The whole query after decompression, in one compiled call. Per touched position i: trees[i] packed split bits;
    fps[i] = (X0, Y0, side, cos lo, sin lo, cos hi, sin hi, th_lo, th_hi, b_lo, b_hi, p, q, beta) with (p, q, beta) of
    h_i; ips[i] = (n_bits, generic, N, band mask kind, has exceptions, exception mask kind, h_i in the family); masks
    as (uint8 bitmap or int64 gap) payloads; side bit bytes; band ranks; exception labels under h_i (stored order).
    Returns (lo, hi, par_lo, par_hi, status): status 0 ok, 1 a label set needs exact arithmetic (the caller then uses
    the general path), 2 a tree bit-count mismatch."""
    k = len(trees)
    N = ips[0][2]
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
        fp = fps[i]
        ip = ips[i]
        D, I, J, S, pos = _decode_tree(trees[i], ip[0], fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8],
                                       fp[9], fp[10], ip[1] != 0)
        if pos != ip[0]:
            return np.zeros(1 << k, np.int64), np.zeros(1 << k, np.int64), 0, 0, np.zeros(1, np.int64), 2
        nl = I.size
        if nl > L:
            L = nl
        nband = 0
        for a in range(nl):
            if S[a] == -1:
                nband += 1
        band_leaves = np.empty(nband, np.int64)
        t = 0
        for a in range(nl):
            if S[a] == -1:
                band_leaves[t] = a
                t += 1
        if ip[3] == 0:
            band_pos = _true_positions(0, bu8s[i], N)
        else:
            band_pos = _true_positions(1, bi64s[i], N)
        n_nb = N - band_pos.size
        if ip[4] != 0:
            if ip[5] == 0:
                exc_nb = _true_positions(0, eu8s[i], n_nb)
            else:
                exc_nb = _true_positions(1, ei64s[i], n_nb)
        else:
            exc_nb = np.zeros(0, np.int64)
        sd, dc, spec, code, exc = _expand_sparse(N, band_pos, exc_nb, sides[i], bids[i], band_leaves, rev)
        side_m[i] = sd
        dec_m[i] = dc
        leaf = np.zeros(nl, np.int8)
        r0 = 0
        r1 = 0
        for a in range(nl):
            if S[a] == -1:
                v = _leaf_label_set(fp[0], fp[1], fp[2], D[a], I[a], J[a], fp[11], fp[12], fp[13])
                if v < 0:
                    return np.zeros(1 << k, np.int64), np.zeros(1 << k, np.int64), 0, 0, np.zeros(1, np.int64), 1
                leaf[a] = v
            elif ip[6] == 0:  # out of the family: a decided region's label set is the union over its leaves
                v = _leaf_label_set(fp[0], fp[1], fp[2], D[a], I[a], J[a], fp[11], fp[12], fp[13])
                if v < 0:
                    return np.zeros(1 << k, np.int64), np.zeros(1 << k, np.int64), 0, 0, np.zeros(1, np.int64), 1
                if S[a] == 0:
                    r0 |= v
                else:
                    r1 |= v
        if ip[6] != 0:
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
        exc_lab[exc_ptr[i]:exc_ptr[i + 1]] = exclabs[i]
    Sx = np.arange(k)
    lo, hi, plo, phi, tup = _query_kernel_sp(side_m, dec_m, N, Sx, reg, leaf_m, spec_ptr, spec_idx, spec_code,
                                             exc_ptr, exc_idx, exc_lab, np.zeros(N, np.bool_))
    return lo, hi, plo, phi, tup, 0


_E_B = np.zeros(0, np.bool_)


def _prep(blobs, h: Discriminator):
    """Decompression and scalar set-up of one touched position for `_query_all` (no decoding). None for the
    single-symbol column layout (handled by the general path)."""
    hd = _HDR.unpack(blobs[0])
    th_lo, th_hi, b_lo, b_hi = hd[7:11]
    N, n_bits, n_exc, flag, dcode = hd[11], hd[12], hd[13], hd[14], hd[15]
    generic = flag >= 4
    flag -= 4 if generic else 0
    pp, qq = unit(h.theta)
    fp = np.array((hd[0], hd[1], hd[2], np.cos(th_lo), np.sin(th_lo), np.cos(th_hi), np.sin(th_hi), th_lo, th_hi, b_lo,
                   b_hi, pp, qq, h.beta), dtype=np.float64)
    tree = _zload(blobs[1])
    tree = np.frombuffer(tree, dtype=np.uint8) if len(tree) else _E_U8
    if flag == 1:  # single symbol column: convert to the mask / side / band-rank form
        sym = np.frombuffer(_zload(blobs[2]), dtype=DTYPES[dcode]).astype(np.int64)
        band = sym >= 3
        nbs = sym[~band]
        bk, bu, bi = 1, _E_U8, np.diff(np.flatnonzero(band), prepend=-1) - 1
        isx = nbs == 2
        ek, eu, ei = 1, _E_U8, np.diff(np.flatnonzero(isx), prepend=-1) - 1
        side_bits = np.packbits(nbs[~isx] == 1)
        side_bits = np.frombuffer(side_bits.tobytes(), dtype=np.uint8) if side_bits.size else _E_U8
        band_ids = sym[band] - 3
        rest = blobs[3:]
    else:
        bk, bu, bi = _mask2(blobs[2])
        j = 3
        if n_exc:
            ek, eu, ei = _mask2(blobs[3])
            j = 4
        else:
            ek, eu, ei = 0, _E_U8, _E_I64
        side_bits = np.frombuffer(_zload(blobs[j]), dtype=np.uint8)
        if side_bits.size == 0:
            side_bits = _E_U8
        band_ids = np.frombuffer(_zload(blobs[j + 1]), dtype=DTYPES[dcode]).astype(np.int64)
        rest = blobs[j + 2:]
    if n_exc:
        fr = Frame.from_array(np.array(hd[3:7]))
        ex, ey = fr.apply(np.frombuffer(_zload(rest[0][1:]), dtype=DTYPES[rest[0][0]]))
        lab = np.ascontiguousarray(h.label(ex, ey), dtype=np.bool_)
    else:
        lab = _E_B
    fam_in = Family(th_lo, th_hi, b_lo, b_hi).contains(h)
    ip = np.array((n_bits, int(generic), N, bk, int(n_exc > 0), ek, int(fam_in)), dtype=np.int64)
    return tree, fp, ip, bu, bi, eu, ei, side_bits, band_ids, lab


class IQFileRef:
    """Reference IQSTORE engine over a persisted container (one file per dataset; groups = positions in
    order). The timed engine is `IQFile` (`iqstore/lean.py`, same answers); this path is its exact fallback and the
    reference of tests/test_e2e.py."""

    def __init__(self, path):
        self.path = path

    def query(self, S: list[int], hs: list[Discriminator]):
        N, groups = read_groups(self.path, S)
        preps = [_prep(g, h) for g, h in zip(groups, hs)]
        if all(p is not None for p in preps):
            lo, hi, plo, phi, tup, st = _query_all(*tuple(zip(*preps)), _REV)
            if st == 0:
                return lo, hi, plo, phi, tup
            if st == 2:
                raise ValueError("tree bit count mismatch")
        return self._query_general(groups, hs, N)

    def _query_general(self, groups, hs, N):
        """Same answer through per-position assembly in Python (single-symbol column layout, or label sets that need
        exact rational arithmetic)."""
        k = len(groups)
        outs = [_decode_query(g, h) for g, h in zip(groups, hs)]
        W = outs[0][0].size
        side = np.empty((k, W), np.uint64)
        dec = np.empty((k, W), np.uint64)
        L = max(o[8] for o in outs)
        reg = np.empty((k, 2), np.int64)
        leaf = np.zeros((k, L), np.int8)
        spec_ptr = np.zeros(k + 1, np.int64)
        exc_ptr = np.zeros(k + 1, np.int64)
        for i, o in enumerate(outs):
            side[i] = o[0]
            dec[i] = o[1]
            reg[i] = o[6]
            leaf[i, :o[8]] = o[7]
            spec_ptr[i + 1] = spec_ptr[i] + o[2].size
            exc_ptr[i + 1] = exc_ptr[i] + o[4].size
        if spec_ptr[-1]:
            spec_idx = np.concatenate([o[2] for o in outs])
            spec_code = np.concatenate([o[3] for o in outs])
        else:
            spec_idx, spec_code = np.zeros(1, np.int64), np.zeros(1, np.int32)
        if exc_ptr[-1]:
            ei = np.concatenate([o[4] for o in outs])
            el = np.concatenate([o[5] for o in outs]).astype(np.bool_)
        else:
            ei, el = np.zeros(1, np.int64), np.zeros(1, np.bool_)
        return _query_kernel_sp(side, dec, N, np.arange(k, dtype=np.int64), reg, leaf, spec_ptr, spec_idx, spec_code,
                                exc_ptr, ei, el, np.zeros(N, np.bool_))


from .lean import IQFile  # noqa: E402,F401  (the timed engine; imported last: lean.py uses the helpers above)
