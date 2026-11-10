"""The IQSTORE per-position representation and exact label sets.

Per position j the partition P_j is a quadtree over a square root box (frame coordinates). Its leaves are HALF-OPEN
rectangles [x0, x1) x [y0, y1). A leaf is
  * decided-0 / decided-1: its CLOSED rectangle is labelled 0 / 1 by every member of the declared family, established by
    a CONSERVATIVE certification test (strict, float-guarded corner test; sound, not an exact characterization: some
    family-decided rectangles stay band leaves);
    the union of decided-b leaves is the family-decided region Omega_j^b;
  * band: anything else.
Per shot s the store keeps only: the side bit b if the shot's leaf is decided (its stored cell is Omega_j^b), the band leaf
id otherwise, or the exact point if the shot is in the exception list E_j.

Label set of a stored cell C under h_{theta,beta}: Lambda = {h(z) : z in C}, the EXACT existential set (real arithmetic on
the float64 coefficients (p, q) = (fl cos theta, fl sin theta) and beta).
  * band leaf (half-open rectangle): exact, see `rect_label_sets` (boundary convention: h = 1[p x + q y > beta]);
  * Omega_j^b: union of the label sets of its decided-b leaves (exact, finite union);
  * exception: {h(x_exact)}.
Every stored point lies in the root box (built from the ingest points), so no cell is unbounded.
"""
from __future__ import annotations

import math
from fractions import Fraction

from dataclasses import dataclass, field

import numpy as np

from .family import Discriminator, Family, exact_sign, unit
from .frame import Frame

# label-set codes: bit0 = 0 is possible, bit1 = 1 is possible
LS_ONLY0 = 1
LS_ONLY1 = 2
LS_BOTH = 3

# per-shot cell codes
CODE_OMEGA0 = 0
CODE_OMEGA1 = 1
CODE_EXCEPTION = 2
BAND_BASE = 3  # band leaf k is stored as BAND_BASE + k


def rect_label_sets(x0, y0, x1, y1, theta: float, beta: float) -> np.ndarray:
    """EXACT label-set codes of half-open rectangles [x0,x1) x [y0,y1) under h = 1[p x + q y > beta], (p, q) = unit(theta).

    1 is possible iff the closed maximum U > beta (the cell is dense in its closure and {> beta} is open).
    0 is possible iff the closed minimum L < beta, or L == beta and the minimizing set meets the cell, i.e. p >= 0 and
    q >= 0 (the cell contains its lower-left corner and its lower / left edges, not the upper / right ones).
    U and L are attained at the corner selected by the signs of p and q; both comparisons use `exact_sign`, so the
    label sets are the exact existential sets (no float guard).
    """
    p, q = unit(theta)
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    xu, yu = (x1 if p > 0 else x0), (y1 if q > 0 else y0)
    xl, yl = (x0 if p >= 0 else x1), (y0 if q >= 0 else y1)
    can1 = exact_sign(xu, yu, p, q, beta) > 0
    sl = exact_sign(xl, yl, p, q, beta)
    can0 = (sl < 0) | ((sl == 0) & (p >= 0) & (q >= 0))  # p, q are scalars
    return (can0.astype(np.int8) * LS_ONLY0) | (can1.astype(np.int8) * LS_ONLY1)


def closed_rect_label_sets(x0, y0, x1, y1, theta: float, beta: float) -> np.ndarray:
    """EXACT label-set codes of CLOSED rectangles [x0,x1] x [y0,y1] under h = 1[p x + q y > beta] (B2 error squares):
    1 possible iff the maximum corner value > beta; 0 possible iff the minimum corner value <= beta (both attained)."""
    p, q = unit(theta)
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    xu, yu = (x1 if p > 0 else x0), (y1 if q > 0 else y0)
    xl, yl = (x0 if p > 0 else x1), (y0 if q > 0 else y1)
    can1 = exact_sign(xu, yu, p, q, beta) > 0
    can0 = exact_sign(xl, yl, p, q, beta) <= 0
    return (can0.astype(np.int8) * LS_ONLY0) | (can1.astype(np.int8) * LS_ONLY1)


LOCATE_MAX = 32  # deepest leaf level `Partition.locate` keys exactly (uint64 (ix << d) | iy)
D_MAX = 40  # deepest leaf level for which every boundary is guaranteed exactly representable (see dyadic_root)
COORD_MAX = 2.0 ** 900  # domain of dyadic_root: |coordinate| <= COORD_MAX (points beyond are stored as exceptions)


def dyadic_root(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Square root box (X0, Y0, side) with power-of-two side and corners that are multiples of side / 2, containing all
    points in its half-open interior.

    Representability: every leaf boundary X0 + i * side / 2^d with d <= D_MAX is a multiple of
    c = side / 2^D_MAX of magnitude <= M + 2 side, where M = max |coordinate|. The side is chosen >= 2^(ceil log2 M - 10),
    so (M + 2 side) / c <= (2^10 + 2) * 2^D_MAX < 2^53 and every such boundary is an exact float64; distinct leaves
    therefore have x0 < x1 and y0 < y1 exactly (never an empty represented cell). Dyadic arithmetic alone is not
    enough: a tiny span far from 0 would collapse boundaries (x = y = 1 gave side 2^-38 and x0 == x1 at depth 16).

    Domain: all coordinates finite with |coordinate| <= COORD_MAX = 2^900. Then side <= 2^902 and
    every boundary has magnitude <= M + 2 side <= 2^904, far inside the float64 range, and c >= side / 2^40 >= 2^-940 is a
    normal power of two (the 2^-900 floor on M keeps side >= 2^-910), so the argument above needs no overflow or
    underflow case. Callers store points outside the domain as exact exceptions (`build_partition`).

    Alignment: X0 = half * floor(min / half) is computed with exact rationals (a float division
    can underflow to -0.0 and exclude a tiny negative point); since side >= mag * 2^-10, |X0| / half <= 2^11 + 1 and
    X0, X0 + side are exact floats. The loop accepts a root only after checking lower AND upper containment of every
    point (X0 <= min, max < X0 + side), doubling the side otherwise."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    lo = min(float(np.min(xs)), float(np.min(ys)))
    hi = max(float(np.max(xs)), float(np.max(ys)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or max(abs(lo), abs(hi)) > COORD_MAX:
        raise ValueError("dyadic_root needs finite coordinates with |coordinate| <= COORD_MAX")
    span = max(hi - lo, 1e-12)
    mag = max(abs(lo), abs(hi), 2.0 ** -900)
    # exact exponents (frexp: v < 2^e): side >= 2 * span and side >= mag * 2^-10, both strictly
    e = max(math.frexp(span)[1] + 1, math.frexp(mag)[1] - 10)
    side = math.ldexp(1.0, e)
    xmin, ymin = float(np.min(xs)), float(np.min(ys))
    xmax, ymax = float(np.max(xs)), float(np.max(ys))
    while True:
        half = side / 2.0
        # X0 = the largest multiple of `half` <= min, computed EXACTLY (a float division can underflow, e.g.
        # -1e-100 / 2^900 -> -0.0, which would exclude the point). |k| <= 2^11 + 1 here.
        X0 = math.floor(Fraction(xmin) / Fraction(half)) * half
        Y0 = math.floor(Fraction(ymin) / Fraction(half)) * half
        # containment of every point in the half-open root box, lower AND upper (all values exact floats)
        if X0 <= xmin and Y0 <= ymin and xmax < X0 + side and ymax < Y0 + side:
            break
        side *= 2.0
    return float(X0), float(Y0), float(side)


@dataclass
class Partition:
    """Quadtree leaves of one position: leaf k is the half-open square of depth[k] at integer position (ix[k], iy[k])
    inside the root box (X0, Y0, side): [X0 + ix*c, X0 + (ix+1)*c) x [Y0 + iy*c, Y0 + (iy+1)*c), c = side / 2^depth."""

    root: tuple[float, float, float]  # (X0, Y0, side), dyadic
    ix: np.ndarray  # int64
    iy: np.ndarray  # int64
    depth: np.ndarray  # int64
    status: np.ndarray  # int8: -1 band, 0 decided-0, 1 decided-1

    @property
    def n_leaves(self) -> int:
        return int(self.ix.size)

    def _cell(self) -> np.ndarray:
        return self.root[2] / (2.0 ** self.depth)

    @property
    def x0(self) -> np.ndarray:
        return self.root[0] + self.ix * self._cell()

    @property
    def x1(self) -> np.ndarray:
        return self.root[0] + (self.ix + 1) * self._cell()

    @property
    def y0(self) -> np.ndarray:
        return self.root[1] + self.iy * self._cell()

    @property
    def y1(self) -> np.ndarray:
        return self.root[1] + (self.iy + 1) * self._cell()

    def locate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Leaf index of each point (half-open leaves, exact comparisons); -1 if outside the root box."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X0, Y0, side = self.root
        out = np.full(x.shape, -1, dtype=np.int64)
        inside = (x >= X0) & (x < X0 + side) & (y >= Y0) & (y < Y0 + side)
        for d in np.unique(self.depth):
            sel = np.flatnonzero(self.depth == d)
            n = 1 << int(d)
            c = side / n
            gx = _grid_index(x, X0, c, n)
            gy = _grid_index(y, Y0, c, n)
            if d > LOCATE_MAX:
                raise ValueError(f"locate supports leaf depth <= {LOCATE_MAX} (uint64 keys)")
            sh = np.uint64(d)
            key_pts = (gx.astype(np.uint64) << sh) | gy.astype(np.uint64)  # exact for d <= 32
            key_leaf = (self.ix[sel].astype(np.uint64) << sh) | self.iy[sel].astype(np.uint64)
            order = np.argsort(key_leaf)
            ks = key_leaf[order]
            pos = np.searchsorted(ks, key_pts)
            pos_c = np.minimum(pos, ks.size - 1)
            hit = inside & (ks[pos_c] == key_pts)
            out[hit] = sel[order[pos_c[hit]]]
        return out


def _grid_index(v: np.ndarray, v0: float, c: float, n: int) -> np.ndarray:
    """floor((v - v0) / c) corrected so that v0 + i*c <= v < v0 + (i+1)*c holds with exact float comparisons."""
    i = np.floor((v - v0) / c).astype(np.int64)
    i = np.clip(i, 0, n - 1)
    lo = v0 + i * c
    i = np.where(v < lo, i - 1, i)
    hi = v0 + (i + 1) * c
    i = np.where(v >= hi, i + 1, i)
    return np.clip(i, 0, n - 1)


@dataclass
class PositionRep:
    """Stored representation of one position of one dataset."""

    frame: Frame
    family: Family
    part: Partition
    codes: np.ndarray  # int32 per shot: CODE_OMEGA0 / CODE_OMEGA1 / CODE_EXCEPTION / BAND_BASE + leaf
    exc_index: np.ndarray  # shot indices in the exception list (sorted)
    exc_x: np.ndarray  # exact frame x of exceptions (float64, derived from the stored raw value)
    exc_y: np.ndarray
    exc_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))  # stored raw values

    def label_set_codes(self, h: Discriminator) -> np.ndarray:
        """Per-shot label-set code under h (LS_ONLY0 / LS_ONLY1 / LS_BOTH)."""
        p = self.part
        leaf_ls = rect_label_sets(p.x0, p.y0, p.x1, p.y1, h.theta, h.beta)
        out = np.empty(self.codes.shape, dtype=np.int8)
        for b, code in ((0, CODE_OMEGA0), (1, CODE_OMEGA1)):
            sel = p.status == b
            ls = np.bitwise_or.reduce(leaf_ls[sel]) if np.any(sel) else 0
            out[self.codes == code] = ls if ls else LS_BOTH  # empty region never referenced; guard
        band = self.codes >= BAND_BASE
        out[band] = leaf_ls[self.codes[band] - BAND_BASE]
        if self.exc_index.size:
            lab = h.label(self.exc_x, self.exc_y)
            out[self.exc_index] = np.where(lab, LS_ONLY1, LS_ONLY0)
        return out


def encode_position(frame: Frame, fam: Family, part: Partition, x: np.ndarray, y: np.ndarray,
                    raw: np.ndarray, exc_mask: np.ndarray) -> PositionRep:
    """Assign each shot its stored cell code (exceptions keep the exact point)."""
    leaf = part.locate(x, y)
    if np.any(leaf < 0) and not np.all(exc_mask[leaf < 0]):
        raise ValueError("point outside the root box and not an exception")
    codes = np.empty(x.shape, dtype=np.int32)
    st = np.where(leaf >= 0, part.status[np.maximum(leaf, 0)], -1)
    codes[st == 0] = CODE_OMEGA0
    codes[st == 1] = CODE_OMEGA1
    band = st == -1
    codes[band] = BAND_BASE + leaf[band]
    exc_index = np.flatnonzero(exc_mask)
    codes[exc_index] = CODE_EXCEPTION
    return PositionRep(frame, fam, part, codes, exc_index, x[exc_index].copy(), y[exc_index].copy(),
                       np.asarray(raw)[exc_index].copy())  # raw dtype kept (lossless)
