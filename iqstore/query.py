"""Separable aggregates and their intervals.

A separable aggregate q = (S, g, h_S): F(X) = sum_s g(l_s), l_s in {0,1}^S the labels of shot s on positions S.
g is given as an array of length 2^|S|; index bit i (value 2^i) is the label of the i-th position of S.

Interval (T1): F- = sum_s min_{l in Lambda_s} g(l), F+ = sum_s max_{l in Lambda_s} g(l), Lambda_s the product of the
per-position label sets. Shots are grouped by their label-set tuple, so the cost is one pass to build keys plus
3^|S| small enumerations.

Arithmetic: every aggregate is accumulated EXACTLY in rationals (each float64 g value is an exact
rational; counts are integers) and returned either as the exact `Fraction` (`exact=True`) or as its correctly rounded
float64. T1 holds exactly for the rational values; because round-to-nearest is monotone, containment F- <= F <= F+
and endpoint attainment (F(X-) == F-, F(X+) == F+) carry over to the rounded floats, provided F(X) is computed by
`exact_value` with the same rounding. Plain float accumulation does not: g = (1e16, -1e16, 1, 0) summed in a
different order gives 0 instead of 1.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

from .family import Discriminator
from .frame import Frame
from .rep import LS_BOTH, LS_ONLY0, LS_ONLY1, PositionRep


def to_float(v: Fraction) -> float:
    """Correctly rounded float64 of an exact rational, with IEEE round-to-nearest overflow semantics: |v| >= DBL_MAX + ulp/2 gives +-inf (float() raises OverflowError instead). Monotone on the extended reals, so
    the rounded interval still contains the rounded value and endpoints stay attained."""
    try:
        return float(v)
    except OverflowError:
        return float("inf") if v > 0 else float("-inf")


def _choices(code: int) -> tuple[int, ...]:
    return {LS_ONLY0: (0,), LS_ONLY1: (1,), LS_BOTH: (0, 1)}[int(code)]


def interval_from_label_sets(ls: np.ndarray, g: np.ndarray, exact: bool = False):
    """ls: (N, k) int8 label-set codes; g: (2^k,) values. Returns (F-, F+): exact Fractions if `exact`, else their
    correctly rounded float64 values."""
    N, k = ls.shape
    if g.shape != (1 << k,):
        raise ValueError("g must have length 2^|S|")
    key = np.zeros(N, dtype=np.int64)
    for i in range(k):
        key = key * 4 + ls[:, i].astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    gq = [Fraction(float(v)) for v in np.asarray(g, dtype=np.float64)]
    lo = Fraction(0)
    hi = Fraction(0)
    for u, c in zip(uk.tolist(), cnt.tolist()):
        codes = []
        v = u
        for _ in range(k):
            codes.append(v % 4)
            v //= 4
        codes = codes[::-1]  # codes[i] belongs to position i of S
        vals = []
        for labs in itertools.product(*[_choices(cd) for cd in codes]):
            idx = sum(b << i for i, b in enumerate(labs))
            vals.append(gq[idx])
        lo += c * min(vals)
        hi += c * max(vals)
    return (lo, hi) if exact else (to_float(lo), to_float(hi))


def intervals_multi(ls: np.ndarray, gs: list[np.ndarray], exact: bool = False) -> list:
    """`interval_from_label_sets` for several g over the same label-set matrix: the shot grouping by label-set tuple is
    done once. Same exact rational arithmetic and rounding."""
    N, k = ls.shape
    key = np.zeros(N, dtype=np.int64)
    for i in range(k):
        key = key * 4 + ls[:, i].astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    groups = []
    for u, c in zip(uk.tolist(), cnt.tolist()):
        codes = []
        v = u
        for _ in range(k):
            codes.append(v % 4)
            v //= 4
        codes = codes[::-1]
        idxs = [sum(b << i for i, b in enumerate(labs)) for labs in itertools.product(*[_choices(cd) for cd in codes])]
        groups.append((c, idxs))
    out = []
    for g in gs:
        g = np.asarray(g, dtype=np.float64)
        if g.shape != (1 << k,):
            raise ValueError("g must have length 2^|S|")
        gq = [Fraction(float(v)) for v in g]
        lo = Fraction(0)
        hi = Fraction(0)
        for c, idxs in groups:
            vals = [gq[i] for i in idxs]
            lo += c * min(vals)
            hi += c * max(vals)
        out.append((lo, hi) if exact else (to_float(lo), to_float(hi)))
    return out


def exact_values_multi(labels: np.ndarray, gs: list[np.ndarray], exact: bool = False) -> list:
    """`exact_value` for several g over the same labels (pattern counts computed once)."""
    k = labels.shape[1]
    idx = np.zeros(labels.shape[0], dtype=np.int64)
    for i in range(k):
        idx |= labels[:, i].astype(np.int64) << i
    cnt = np.bincount(idx, minlength=1 << k).tolist()
    out = []
    for g in gs:
        gv = np.asarray(g, dtype=np.float64)
        v = sum((int(c) * Fraction(float(gv[i])) for i, c in enumerate(cnt) if c), Fraction(0))
        out.append(v if exact else to_float(v))
    return out


def _dyadic(g: np.ndarray) -> tuple[list[int], int]:
    """Every float64 of g as an integer numerator over one common power-of-two denominator D (exact)."""
    rs = [float(v).as_integer_ratio() for v in np.asarray(g, dtype=np.float64)]
    D = max(d for _, d in rs)
    return [n * (D // d) for n, d in rs], D


def exact_dot(counts, g: np.ndarray, exact: bool = False):
    """sum_i counts[i] * g[i] (integer counts, float64 g) by exact integer arithmetic; the same value as the Fraction
    accumulation of `exact_value`, correctly rounded (or the Fraction if `exact`). Used by every engine's completion of
    a mitigated expectation (identical work per method)."""
    num, D = _dyadic(g)
    v = Fraction(sum(int(c) * n for c, n in zip(counts, num) if c), D)
    return v if exact else to_float(v)


def interval_from_tuples(tup: np.ndarray, k: int, g: np.ndarray, exact: bool = False):
    """(F-, F+) of T1 from the number of shots per label-set tuple (`tup[sum_i code_i 4^i]`, code 1 = {0}, 2 = {1},
    3 = {0, 1}; the 5th output of the query kernels): the same exact values as `interval_from_label_sets` on the
    label-set matrix those counts summarize."""
    if len(g) != 1 << k:
        raise ValueError("g must have length 2^|S|")
    num, D = _dyadic(g)
    lo = 0
    hi = 0
    for key in np.flatnonzero(tup).tolist():
        c = int(tup[key])
        idxs = [0]
        for i in range(k):
            code = (key >> (2 * i)) & 3
            if code == 1:
                continue
            if code == 2:
                idxs = [x | (1 << i) for x in idxs]
            else:
                idxs = idxs + [x | (1 << i) for x in idxs]
        vals = [num[x] for x in idxs]
        lo += c * min(vals)
        hi += c * max(vals)
    lo, hi = Fraction(lo, D), Fraction(hi, D)
    return (lo, hi) if exact else (to_float(lo), to_float(hi))


def interval(reps: list[PositionRep], hs: list[Discriminator], g: np.ndarray, exact: bool = False):
    ls = np.column_stack([r.label_set_codes(h) for r, h in zip(reps, hs)])
    return interval_from_label_sets(ls, np.asarray(g, dtype=np.float64), exact=exact)


def exact_labels(raw_cols: list[np.ndarray], frames: list[Frame], hs: list[Discriminator]) -> np.ndarray:
    """Exact labels (N, k) from raw complex IQ (re-classification of the raw data captured at ingest)."""
    out = []
    for z, f, h in zip(raw_cols, frames, hs):
        x, y = f.apply(z)
        out.append(h.label(x, y))
    return np.column_stack(out).astype(np.int8)


def exact_value(labels: np.ndarray, g: np.ndarray, exact: bool = False):
    """F(X) = sum_s g(l_s) accumulated exactly (pattern counts x rational g); Fraction if `exact`, else correctly rounded."""
    k = labels.shape[1]
    idx = np.zeros(labels.shape[0], dtype=np.int64)
    for i in range(k):
        idx |= labels[:, i].astype(np.int64) << i
    cnt = np.bincount(idx, minlength=1 << k)
    gv = np.asarray(g, dtype=np.float64)
    v = sum((int(c) * Fraction(float(gv[i])) for i, c in enumerate(cnt.tolist()) if c), Fraction(0))
    return v if exact else to_float(v)


# ---- standard g functions -------------------------------------------------------------------------------------------

def g_count(k: int, pattern: int) -> np.ndarray:
    g = np.zeros(1 << k)
    g[pattern] = 1.0
    return g


def g_parity(k: int) -> np.ndarray:
    """(-1)^{xor of labels}; divide the sum by N for <Z_S>."""
    return np.array([(-1.0) ** bin(i).count("1") for i in range(1 << k)])
