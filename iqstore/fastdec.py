"""Compiled corner test and quadtree decoder (prereg section 6: decode is inside every timed run).

`rect_decided_nb` evaluates the same expression as `family.rect_decided` (T2 corner test with the 2k*pi shift and the
sound-side rounding guard) one rectangle at a time, with the family's cos / sin taken from NumPy once. The encoder
(`storage._quadtree_bits`) and the decoder (`decode_tree`) both use it, so the decoder always reproduces the encoder's
tree; the encoder also checks it against the construction's leaf status (computed with `family.rect_decided`), so any
disagreement between the two implementations stops ingest instead of passing silently.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .family import FLOAT_GUARD, TWO_PI

_PI = float(np.pi)


@njit(cache=True)
def _min_dot(x, y, cl, sl, ch, sh, th_lo, th_hi):
    r = math.hypot(x, y)
    phi = math.atan2(y, x)
    e_lo = cl * x + sl * y
    e_hi = ch * x + sh * y
    ends = e_lo if e_lo < e_hi else e_hi
    a = phi + _PI
    k = math.ceil((th_lo - a) / TWO_PI)
    if (a + k * TWO_PI) <= th_hi:
        return -r if -r < ends else ends
    return ends


@njit(cache=True)
def _max_dot(x, y, cl, sl, ch, sh, th_lo, th_hi):
    r = math.hypot(x, y)
    phi = math.atan2(y, x)
    e_lo = cl * x + sl * y
    e_hi = ch * x + sh * y
    ends = e_lo if e_lo > e_hi else e_hi
    k = math.ceil((th_lo - phi) / TWO_PI)
    if (phi + k * TWO_PI) <= th_hi:
        return r if r > ends else ends
    return ends


_SEP = 1e-9  # direction test margin (radians, as |sin| of the angle to an end of the range)


@njit(cache=True)
def _minmax_dot(x, y, cl, sl, ch, sh, th_lo, th_hi):
    """(_min_dot, _max_dot) of `family.minmax_dot_over_angles` for one point, with identical results.

    The two direction tests ("some phi + 2k pi in [th_lo, th_hi]" for w and for -w) are first decided by cross products
    with the end directions (exact up to ~1e-15 relative): a direction farther than _SEP (as |sin|) from both ends is
    classified without atan2 / hypot, and its answer equals that of the atan2 test, whose own rounding error is ~1e-15
    rad. Within _SEP of an end, the original atan2 expression is evaluated, so the result is the reference result."""
    e_lo = cl * x + sl * y
    e_hi = ch * x + sh * y
    mn = e_lo if e_lo < e_hi else e_hi
    mx = e_lo if e_lo > e_hi else e_hi
    if x == 0.0 and y == 0.0:
        return mn, mx  # r = 0 and every end product is 0: the extremes are 0 either way
    tol = _SEP * (abs(x) + abs(y))
    c1 = cl * y - sl * x  # sin(phi - th_lo) |w|  (direction w)
    c2 = sh * x - ch * y  # sin(th_hi - phi) |w|
    # direction w (maximum): inside iff c1 >= 0 and c2 >= 0 (range narrower than pi)
    if c1 < -tol or c2 < -tol:
        in_w = 0
    elif c1 > tol and c2 > tol:
        in_w = 1
    else:
        in_w = -1
    # direction -w (minimum): the cross products change sign
    if -c1 < -tol or -c2 < -tol:
        in_m = 0
    elif -c1 > tol and -c2 > tol:
        in_m = 1
    else:
        in_m = -1
    if in_w == 0 and in_m == 0:
        return mn, mx
    r = math.hypot(x, y)
    if in_w == -1 or in_m == -1:
        phi = math.atan2(y, x)
        if in_m == -1:
            a = phi + _PI
            k = math.ceil((th_lo - a) / TWO_PI)
            in_m = 1 if (a + k * TWO_PI) <= th_hi else 0
        if in_w == -1:
            k = math.ceil((th_lo - phi) / TWO_PI)
            in_w = 1 if (phi + k * TWO_PI) <= th_hi else 0
    if in_m == 1 and -r < mn:
        mn = -r
    if in_w == 1 and r > mx:
        mx = r
    return mn, mx


@njit(cache=True)
def _minmax_dot_ref(x, y, cl, sl, ch, sh, th_lo, th_hi):
    """The atan2 form (reference for tests)."""
    r = math.hypot(x, y)
    phi = math.atan2(y, x)
    e_lo = cl * x + sl * y
    e_hi = ch * x + sh * y
    mn = e_lo if e_lo < e_hi else e_hi
    mx = e_lo if e_lo > e_hi else e_hi
    a = phi + _PI
    k = math.ceil((th_lo - a) / TWO_PI)
    if (a + k * TWO_PI) <= th_hi:
        if -r < mn:
            mn = -r
    k = math.ceil((th_lo - phi) / TWO_PI)
    if (phi + k * TWO_PI) <= th_hi:
        if r > mx:
            mx = r
    return mn, mx


@njit(cache=True)
def _rect_status(x0, y0, x1, y1, cl, sl, ch, sh, th_lo, th_hi, b_lo, b_hi):
    """Status of `family.rect_decided` (corners in its order). The running min only decreases and the running max only
    increases, so once neither side can still be decided the status is -1 and the remaining corners are skipped."""
    sc = max(abs(x0), abs(x1)) + max(abs(y0), abs(y1)) + max(abs(b_lo), abs(b_hi))
    g = FLOAT_GUARD * (sc + 1.0)
    t1 = b_hi + g
    t0 = b_lo - g
    mn, mx = _minmax_dot(x0, y0, cl, sl, ch, sh, th_lo, th_hi)
    if mn <= t1 and mx >= t0:
        return -1
    for cx, cy in ((x1, y0), (x1, y1), (x0, y1)):
        a, b = _minmax_dot(cx, cy, cl, sl, ch, sh, th_lo, th_hi)
        if a < mn:
            mn = a
        if b > mx:
            mx = b
        if mn <= t1 and mx >= t0:
            return -1
    out = -1
    if mn > t1:
        out = 1
    if mx < t0:
        out = 0
    return out


@njit(cache=True)
def _node_status(X0, Y0, side, d, i, j, cl, sl, ch, sh, th_lo, th_hi, b_lo, b_hi):
    c = side / (2.0 ** d)
    return _rect_status(X0 + i * c, Y0 + j * c, X0 + (i + 1) * c, Y0 + (j + 1) * c, cl, sl, ch, sh, th_lo, th_hi,
                        b_lo, b_hi)


@njit(cache=True)
def node_status_arr(X0, Y0, side, D, I, J, cl, sl, ch, sh, th_lo, th_hi, b_lo, b_hi):
    out = np.empty(I.size, dtype=np.int8)
    for k in range(I.size):
        out[k] = _node_status(X0, Y0, side, D[k], I[k], J[k], cl, sl, ch, sh, th_lo, th_hi, b_lo, b_hi)
    return out


def fam_args(fam):
    """Family scalars in the order the kernels take them; cos / sin from NumPy, as in `family.min_dot_over_angles`."""
    return (float(np.cos(fam.th_lo)), float(np.sin(fam.th_lo)), float(np.cos(fam.th_hi)), float(np.sin(fam.th_hi)),
            float(fam.th_lo), float(fam.th_hi), float(fam.b_lo), float(fam.b_hi))


@njit(cache=True)
def _decode_tree(packed, n_bits, X0, Y0, side, cl, sl, ch, sh, th_lo, th_hi, b_lo, b_hi, generic):
    """Level-by-level decode of the MSB-first packed split bits; nodes of a level in (ix, iy) order. Returns leaves
    (depth, ix, iy, status) in (depth, ix, iy) order and the number of bits consumed (-1 if the stream ran out)."""
    cap = 1024
    ld = np.empty(cap, np.int64)
    li = np.empty(cap, np.int64)
    lj = np.empty(cap, np.int64)
    ls = np.empty(cap, np.int8)
    nl = 0
    I = np.zeros(1, np.int64)
    J = np.zeros(1, np.int64)
    d = 0
    pos = 0
    while I.size > 0:
        nsplit = 0
        split = np.zeros(I.size, np.bool_)
        c = side / (2.0 ** d)  # the cell size of `_node_status`, computed once per level (same value)
        for k in range(I.size):
            st = _rect_status(X0 + I[k] * c, Y0 + J[k] * c, X0 + (I[k] + 1) * c, Y0 + (J[k] + 1) * c, cl, sl, ch, sh,
                              th_lo, th_hi, b_lo, b_hi)
            if st >= 0 and not generic:
                b = 0
            else:
                if pos >= n_bits:
                    return ld[:0], li[:0], lj[:0], ls[:0], -1
                b = (packed[pos >> 3] >> (7 - (pos & 7))) & 1
                pos += 1
            if b:
                split[k] = True
                nsplit += 1
            else:
                if nl == cap:
                    cap *= 2
                    ld2 = np.empty(cap, np.int64)
                    li2 = np.empty(cap, np.int64)
                    lj2 = np.empty(cap, np.int64)
                    ls2 = np.empty(cap, np.int8)
                    ld2[:nl] = ld[:nl]
                    li2[:nl] = li[:nl]
                    lj2[:nl] = lj[:nl]
                    ls2[:nl] = ls[:nl]
                    ld, li, lj, ls = ld2, li2, lj2, ls2
                ld[nl] = d
                li[nl] = I[k]
                lj[nl] = J[k]
                ls[nl] = st
                nl += 1
        I2 = np.empty(4 * nsplit, np.int64)
        J2 = np.empty(4 * nsplit, np.int64)
        # children in (ix, iy) order without sorting: the level is sorted by (ix, iy); for each run of parents with the
        # same ix, emit row 2*ix (columns 2*iy, 2*iy+1 in iy order), then row 2*ix+1
        t = 0
        a = 0
        while a < I.size:
            e = a
            while e < I.size and I[e] == I[a]:
                e += 1
            for di in range(2):
                for k in range(a, e):
                    if split[k]:
                        for dj in range(2):
                            I2[t] = 2 * I[k] + di
                            J2[t] = 2 * J[k] + dj
                            t += 1
            a = e
        I = I2
        J = J2
        d += 1
    return ld[:nl], li[:nl], lj[:nl], ls[:nl], pos


def decode_tree(packed: bytes, n_bits: int, root, fam, generic: bool = False):
    """(depth, ix, iy, status) of the leaves in (depth, ix, iy) order; status from the same kernel as the encoder."""
    buf = np.frombuffer(packed, dtype=np.uint8) if len(packed) else np.zeros(1, np.uint8)
    D, I, J, S, pos = _decode_tree(buf, n_bits, float(root[0]), float(root[1]), float(root[2]), *fam_args(fam),
                                   bool(generic))
    if pos != n_bits:
        raise ValueError("tree bit count mismatch")
    # leaves are emitted level by level, each level in (ix, iy) order: already (depth, ix, iy) order
    return D, I, J, S
