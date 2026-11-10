"""Linear discriminators, declared families and the family-decided test.

All coordinates are frame coordinates (x, y) = T_j z. A discriminator is h_{theta,beta}(x, y) = 1[cos(theta) x +
sin(theta) y > beta] (strict '>'; points exactly on the line are labelled 0). A family is the box
theta in [th_lo, th_hi], beta in [b_lo, b_hi] with th_hi - th_lo < pi.

T2 (DERIVED; checked in D2): the set labelled 1 by every member is convex (intersection of open half-planes), the set
labelled 0 by every member is convex (intersection of closed half-planes); so a closed rectangle is decided iff its four
corners are. The single-point test uses the exact extremum of r cos(theta - phi) over the angle interval, with the 2k*pi
shift (a test without the shift wrongly certifies w = (-1, 0) for theta in [-0.1, 0.1]).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi
FLOAT_GUARD = 8.0 * np.finfo(np.float64).eps  # relative guard for float rounding of <u, z> (sound side only)


def guard(scale) -> np.ndarray:
    """Absolute rounding guard for comparisons of <u, z> against beta; `scale` ~ |x| + |y| + |beta|."""
    return FLOAT_GUARD * (np.asarray(scale, dtype=np.float64) + 1.0)


_EPS = np.finfo(np.float64).eps  # 2^-52


def unit(theta: float) -> tuple[float, float]:
    """The discriminator's coefficients, fixed as float64: (p, q) = (fl(cos theta), fl(sin theta)). Every label and
    label set is defined EXACTLY (real arithmetic) in terms of these floats."""
    return float(np.cos(theta)), float(np.sin(theta))


def exact_sign(x, y, p: float, q: float, beta) -> np.ndarray:
    """sign(p*x + q*y - beta) in EXACT real arithmetic for float64 inputs (int8 in {-1, 0, 1}).

    Float filter: d = fl(fl(fl(p x) + fl(q y)) - beta) has |d - d_exact| <= 2 eps (|p x| + |q y| + |beta|) (three
    roundings, each <= eps/2 relative to a term bounded by that sum, with margin); outside the band the float sign is
    exact. Inside it the sign is computed with exact rationals (fractions.Fraction of the floats)."""
    from fractions import Fraction

    x, y, b = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
                                  np.asarray(beta, dtype=np.float64))
    shape = x.shape
    x, y, b = x.ravel(), y.ravel(), b.ravel()
    px, qy = p * x, q * y
    d = (px + qy) - b
    err = 2.0 * _EPS * (np.abs(px) + np.abs(qy) + np.abs(b)) + np.finfo(np.float64).tiny
    out = np.sign(d).astype(np.int8)
    unsure = np.flatnonzero(~(np.abs(d) > err))  # also catches NaN-free exact ties
    if unsure.size:
        fp, fq = Fraction(float(p)), Fraction(float(q))
        for i in unsure:
            v = fp * Fraction(float(x[i])) + fq * Fraction(float(y[i])) - Fraction(float(b[i]))
            out[i] = (v > 0) - (v < 0)
    return out.reshape(shape)


@dataclass(frozen=True)
class Discriminator:
    theta: float
    beta: float

    def label(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """h(x, y) = 1[p x + q y > beta], evaluated exactly (see `exact_sign`)."""
        p, q = unit(self.theta)
        return exact_sign(x, y, p, q, self.beta) > 0


@dataclass(frozen=True)
class Family:
    th_lo: float
    th_hi: float
    b_lo: float
    b_hi: float

    def __post_init__(self) -> None:
        if not (self.th_hi >= self.th_lo and self.th_hi - self.th_lo < np.pi):
            raise ValueError("family angle range must satisfy 0 <= th_hi - th_lo < pi")
        if not self.b_hi >= self.b_lo:
            raise ValueError("family offset range must satisfy b_lo <= b_hi")

    def contains(self, h: Discriminator) -> bool:
        # angle compared modulo 2*pi
        t = h.theta
        k = np.ceil((self.th_lo - t) / TWO_PI)
        t2 = t + k * TWO_PI
        return bool(t2 <= self.th_hi and self.b_lo <= h.beta <= self.b_hi)

    def to_array(self) -> np.ndarray:
        return np.array([self.th_lo, self.th_hi, self.b_lo, self.b_hi], dtype=np.float64)


def _angle_in_interval(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True where some a + 2k*pi lies in [lo, hi] (hi - lo < 2*pi)."""
    k = np.ceil((lo - a) / TWO_PI)
    return (a + k * TWO_PI) <= hi


def min_dot_over_angles(x: np.ndarray, y: np.ndarray, th_lo: float, th_hi: float) -> np.ndarray:
    """Exact min over theta in [th_lo, th_hi] of cos(theta) x + sin(theta) y (vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    ends = np.minimum(np.cos(th_lo) * x + np.sin(th_lo) * y, np.cos(th_hi) * x + np.sin(th_hi) * y)
    interior = _angle_in_interval(phi + np.pi, th_lo, th_hi)
    return np.where(interior, np.minimum(-r, ends), ends)


def max_dot_over_angles(x: np.ndarray, y: np.ndarray, th_lo: float, th_hi: float) -> np.ndarray:
    """Exact max over theta in [th_lo, th_hi] of cos(theta) x + sin(theta) y (vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    ends = np.maximum(np.cos(th_lo) * x + np.sin(th_lo) * y, np.cos(th_hi) * x + np.sin(th_hi) * y)
    interior = _angle_in_interval(phi, th_lo, th_hi)
    return np.where(interior, np.maximum(r, ends), ends)


def minmax_dot_over_angles(x: np.ndarray, y: np.ndarray, th_lo: float, th_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """(min_dot_over_angles, max_dot_over_angles) in one pass: the same elementwise float operations, with the shared
    r = hypot(x, y), phi = arctan2(y, x) and endpoint products computed once (identical results)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    e_lo = np.cos(th_lo) * x + np.sin(th_lo) * y
    e_hi = np.cos(th_hi) * x + np.sin(th_hi) * y
    mn = np.minimum(e_lo, e_hi)
    mx = np.maximum(e_lo, e_hi)
    mn = np.where(_angle_in_interval(phi + np.pi, th_lo, th_hi), np.minimum(-r, mn), mn)
    mx = np.where(_angle_in_interval(phi, th_lo, th_hi), np.maximum(r, mx), mx)
    return mn, mx


def point_decided(x: np.ndarray, y: np.ndarray, fam: Family) -> np.ndarray:
    """-1 = not decided, 0 = labelled 0 by every member, 1 = labelled 1 by every member."""
    mn = min_dot_over_angles(x, y, fam.th_lo, fam.th_hi)
    mx = max_dot_over_angles(x, y, fam.th_lo, fam.th_hi)
    g = guard(np.abs(x) + np.abs(y) + max(abs(fam.b_lo), abs(fam.b_hi)))
    out = np.full(np.shape(x), -1, dtype=np.int8)
    out[mn > fam.b_hi + g] = 1
    out[mx < fam.b_lo - g] = 0
    return out


def rect_decided(x0, y0, x1, y1, th_lo: float, th_hi: float, b_lo: float, b_hi: float) -> np.ndarray:
    """Decided status of CLOSED rectangles [x0,x1]x[y0,y1] for the family box (corner test, T2).

    Returns int8 array: 1 if every member labels every point 1, 0 if every member labels every point 0, else -1.
    Sound for half-open cells (a half-open cell is a subset of its closed rectangle). A float-rounding guard is
    applied on the sound side (a rectangle within the guard of a member's boundary is reported undecided).
    """
    xs = (x0, x1, x1, x0)
    ys = (y0, y0, y1, y1)
    mn = None
    mx = None
    for cx, cy in zip(xs, ys):
        a = min_dot_over_angles(cx, cy, th_lo, th_hi)
        b = max_dot_over_angles(cx, cy, th_lo, th_hi)
        mn = a if mn is None else np.minimum(mn, a)
        mx = b if mx is None else np.maximum(mx, b)
    sc = np.maximum(np.abs(x0), np.abs(x1)) + np.maximum(np.abs(y0), np.abs(y1)) + max(abs(b_lo), abs(b_hi))
    g = guard(sc)
    out = np.full(np.shape(mn), -1, dtype=np.int8)
    out[mn > b_hi + g] = 1
    out[mx < b_lo - g] = 0  # strict: consistent with the guarded label sets of rep.rect_label_sets
    return out


def fit_lda(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> Discriminator:
    """Two-class LDA (pooled covariance, equal priors) in frame coordinates, returned as (theta, beta)."""
    a = np.column_stack([x0, y0])
    b = np.column_stack([x1, y1])
    m0 = a.mean(axis=0)
    m1 = b.mean(axis=0)
    s = (np.cov(a, rowvar=False) * (len(a) - 1) + np.cov(b, rowvar=False) * (len(b) - 1)) / (len(a) + len(b) - 2)
    w = np.linalg.solve(s, m1 - m0)
    c = 0.5 * float(w @ (m0 + m1))
    n = float(np.hypot(w[0], w[1]))
    return Discriminator(float(np.arctan2(w[1], w[0])), c / n)


def family_from_widths(ref: Discriminator, d_theta: float, d_beta: float) -> Family:
    return Family(ref.theta - d_theta, ref.theta + d_theta, ref.beta - d_beta, ref.beta + d_beta)


def bootstrap_widths(x0, y0, x1, y1, n_boot: int, seed: int, q: float = 0.99) -> tuple[float, float, Discriminator]:
    """q-quantile of |theta_b - theta_ref| and |beta_b - beta_ref| over bootstrap refits."""
    ref = fit_lda(x0, y0, x1, y1)
    rng = np.random.default_rng(seed)
    dt = np.empty(n_boot)
    db = np.empty(n_boot)
    for i in range(n_boot):
        i0 = rng.integers(0, len(x0), len(x0))
        i1 = rng.integers(0, len(x1), len(x1))
        h = fit_lda(x0[i0], y0[i0], x1[i1], y1[i1])
        dt[i] = abs((h.theta - ref.theta + np.pi) % TWO_PI - np.pi)
        db[i] = abs(h.beta - ref.beta)
    return float(np.quantile(dt, q)), float(np.quantile(db, q)), ref
