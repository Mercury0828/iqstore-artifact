"""Phase-0 prototype of partition construction and its worst-case certifier.

Certifier (branch and bound over the family box; sound). For a sub-box B = [ta, tb] x [ba, bb] of the family, a stored
rectangle can be crossed by some member of B (label set {0, 1}) only if its CLOSED rectangle is not decided by B (T2
corner test on B). Hence for every h in B, U(h) <= UB(B) := total count of rectangles not decided by B. Starting from a
K0 x K0 grid of sub-boxes, a sub-box with UB(B) <= budget is certified; otherwise the exact U(h_c) at its centre is
computed (exact label sets): if U(h_c) > budget the family contains a violating member (returned as witness); else the
sub-box is split in four, down to `max_level`. If every leaf sub-box is certified, max_h U(h) <= max UB <= budget.
The certifier is shared by IQSTORE, B2 (per-shot error squares) and B4q (grid cells): equal certificate strength.

Construction (prototype, certified-feasible greedy, NO optimality claim): uniform grid at depth d0 over a dyadic root
box; leaves decided by the full family are marked; while the certifier fails, the heaviest leaves among those crossed by
the witness (or undecided in the uncertified sub-box) are split; leaves already at max depth move their shots to the
exception list (stored exactly). Afterwards sibling leaves are merged when this cannot raise any U(h). Terminates with a
certificate (the exception list is the safety net).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .family import Discriminator, Family, guard, minmax_dot_over_angles, rect_decided
from .rep import COORD_MAX, D_MAX, LOCATE_MAX, Partition, dyadic_root, rect_label_sets


@dataclass
class CertResult:
    ok: bool  # True: max over the family of U(h) <= budget is certified
    upper: int  # max UB over certified leaf sub-boxes (sound upper bound when ok)
    lower: int  # max exact U(h) over evaluated members (a true lower bound on max_h U(h))
    witness: Discriminator | None  # member with exact U > budget (if found)
    hot: np.ndarray | None  # indices of rectangles to refine (crossed by witness, or undecided in the stuck box)
    boxes: int  # sub-boxes visited


def _undecided(x0, y0, x1, y1, ta, tb, ba, bb) -> np.ndarray:
    """True where the closed rectangle is NOT decided by every member of the box. The exclusion carries the same
    float guard as `rect_label_sets` (a rectangle is excluded only if, for every member, the guarded label set is a
    singleton), so a certified bound always bounds the uncertainty the query path reports."""
    n = np.size(x0)
    # the four corners in one batch (elementwise operations: identical values; min / max are exact)
    lo, hi = minmax_dot_over_angles(np.concatenate([x0, x1, x1, x0]), np.concatenate([y0, y0, y1, y1]), ta, tb)
    mn = lo.reshape(4, n).min(axis=0)
    mx = hi.reshape(4, n).max(axis=0)
    # guard of rect_label_sets at the largest |beta| of the box (monotone in |beta|, so it covers every member)
    g = guard(np.maximum(np.abs(x0), np.abs(x1)) + np.maximum(np.abs(y0), np.abs(y1)) + max(abs(ba), abs(bb)))
    return ~((mn > bb + g) | (mx < ba - g))


def certify(x0, y0, x1, y1, counts, fam: Family, budget: int, K0: int = 8, max_level: int = 12,
            max_boxes: int = 400000) -> CertResult:
    x0, y0, x1, y1 = (np.asarray(a, dtype=np.float64) for a in (x0, y0, x1, y1))
    c = np.asarray(counts).astype(np.int64)
    live = np.flatnonzero(c > 0)
    if live.size == 0 or int(c.sum()) <= budget:
        return CertResult(True, int(c.sum()) if live.size else 0, 0, None, None, 0)
    th = np.linspace(fam.th_lo, fam.th_hi, K0 + 1)
    be = np.linspace(fam.b_lo, fam.b_hi, K0 + 1)
    stack = [(th[i], th[i + 1], be[j], be[j + 1], 0, live) for i in range(K0) for j in range(K0)]
    upper = 0
    lower = 0
    boxes = 0
    while stack:
        ta, tb, ba, bb, lev, idx = stack.pop()
        boxes += 1
        und = _undecided(x0[idx], y0[idx], x1[idx], y1[idx], ta, tb, ba, bb)
        sub = idx[und]
        ub = int(c[sub].sum())
        if ub <= budget:
            upper = max(upper, ub)
            continue
        hc = Discriminator(0.5 * (ta + tb), 0.5 * (ba + bb))
        ls = rect_label_sets(x0[sub], y0[sub], x1[sub], y1[sub], hc.theta, hc.beta)
        crossed = sub[ls == 3]
        u = int(c[crossed].sum())
        lower = max(lower, u)
        if u > budget:
            return CertResult(False, ub, lower, hc, crossed, boxes)
        if lev >= max_level or boxes >= max_boxes:
            return CertResult(False, ub, lower, None, sub, boxes)
        tm = 0.5 * (ta + tb)
        bm = 0.5 * (ba + bb)
        for (p, q) in ((ta, tm), (tm, tb)):
            for (r, s) in ((ba, bm), (bm, bb)):
                stack.append((p, q, r, s, lev + 1, sub))
    return CertResult(True, upper, lower, None, None, boxes)


def sampled_lower(x0, y0, x1, y1, counts, fam: Family, n: int = 33) -> tuple[int, Discriminator]:
    """Exact U(h) maximized over an n x n grid of family members (a lower bound on max_h U(h))."""
    c = np.asarray(counts).astype(np.int64)
    best, bh = 0, Discriminator(fam.th_lo, fam.b_lo)
    for t in np.linspace(fam.th_lo, fam.th_hi, n):
        for b in np.linspace(fam.b_lo, fam.b_hi, n):
            ls = rect_label_sets(x0, y0, x1, y1, float(t), float(b))
            u = int(c[ls == 3].sum())
            if u > best:
                best, bh = u, Discriminator(float(t), float(b))
    return best, bh


def _band_rects(part: Partition, counts: np.ndarray):
    band = (part.status == -1) & (counts > 0)
    idx = np.flatnonzero(band)
    return idx, part.x0[idx], part.y0[idx], part.x1[idx], part.y1[idx], counts[idx]


def certify_partition(part: Partition, counts: np.ndarray, fam: Family, budget: int) -> tuple[CertResult, np.ndarray]:
    idx, a, b, cc, d, cnt = _band_rects(part, counts)
    res = certify(a, b, cc, d, cnt, fam, budget)
    return res, idx


def _counts(part: Partition, leaf_of_point: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.bincount(leaf_of_point[active], minlength=part.n_leaves)


def _with_status(part: Partition, fam: Family) -> Partition:
    part.status = rect_decided(part.x0, part.y0, part.x1, part.y1, fam.th_lo, fam.th_hi, fam.b_lo, fam.b_hi)
    return part


def build_partition(x: np.ndarray, y: np.ndarray, fam: Family, tau: float, d0: int = 3, max_depth: int = 16,
                    split_frac: float = 0.5, max_iter: int = 2000):
    """Return (partition, exception mask, CertResult, iterations). Certified: max_h U(h) <= floor(tau * N)."""
    N = x.size
    if not (0 <= d0 <= max_depth <= min(D_MAX, LOCATE_MAX)):
        raise ValueError(f"need 0 <= d0 <= max_depth <= {min(D_MAX, LOCATE_MAX)} (exact boundaries, exact locate keys)")
    budget = int(np.floor(tau * N))
    # points outside the representability domain of dyadic_root are stored exactly (exceptions)
    out_dom = ~((np.abs(x) <= COORD_MAX) & (np.abs(y) <= COORD_MAX))
    if np.all(out_dom):
        X0, Y0, side = 0.0, 0.0, 1.0
    else:
        X0, Y0, side = dyadic_root(x[~out_dom], y[~out_dom])
    n = 1 << d0
    gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    part = _with_status(Partition((X0, Y0, side), gx.ravel().astype(np.int64), gy.ravel().astype(np.int64),
                                  np.full(n * n, d0, dtype=np.int64), np.zeros(n * n, dtype=np.int8)), fam)
    exc = out_dom.copy()
    it = 0
    while True:
        leaf = part.locate(x, y)
        cnt = _counts(part, leaf, ~exc)
        res, idx = certify_partition(part, cnt, fam, budget)
        if res.ok:
            break
        if it >= max_iter:
            # safety net: every shot in a family-undecided leaf becomes an exception (certifies trivially)
            exc |= part.status[leaf] == -1
            break
        it += 1
        hot = idx[res.hot]
        order = hot[np.argsort(-cnt[hot], kind="stable")]
        take = order[: max(1, int(np.ceil(split_frac * order.size)))]
        at_max = take[part.depth[take] >= max_depth]
        if at_max.size:
            exc |= np.isin(leaf, at_max)
        to_split = take[part.depth[take] < max_depth]
        if to_split.size:
            keep = np.ones(part.n_leaves, dtype=bool)
            keep[to_split] = False
            cix, ciy, cd = [part.ix[keep]], [part.iy[keep]], [part.depth[keep]]
            for dx in (0, 1):
                for dy in (0, 1):
                    cix.append(2 * part.ix[to_split] + dx)
                    ciy.append(2 * part.iy[to_split] + dy)
                    cd.append(part.depth[to_split] + 1)
            part = _with_status(Partition(part.root, np.concatenate(cix), np.concatenate(ciy), np.concatenate(cd),
                                          np.zeros(sum(a.size for a in cix), dtype=np.int8)), fam)
    part = prune(part, _counts(part, part.locate(x, y), ~exc), fam)
    leaf = part.locate(x, y)
    cnt = _counts(part, leaf, ~exc)
    res, _ = certify_partition(part, cnt, fam, budget)
    if not res.ok:  # cannot happen after the safety net unless pruning broke something -> hard error
        raise AssertionError("partition not certified after construction")
    if not (np.all(part.x0 < part.x1) and np.all(part.y0 < part.y1)):
        raise AssertionError("empty represented leaf (boundary not exactly representable)")
    return part, exc, res, it


def prune(part: Partition, counts: np.ndarray, fam: Family) -> Partition:
    """Merge four sibling leaves into their parent when this cannot raise any U(h): (a) all four decided with the same
    side (the parent is then decided by the same corner test), or (b) all four hold no non-exception shot (the merged
    leaf is empty; its status is recomputed by the corner test). Repeated bottom-up until nothing merges."""
    while True:
        key = {(int(d), int(i), int(j)): k for k, (d, i, j) in enumerate(zip(part.depth, part.ix, part.iy))}
        groups: dict = {}
        for (d, i, j), k in key.items():
            if d > 0:
                groups.setdefault((d - 1, i >> 1, j >> 1), []).append(k)
        merge = []
        for parent, ks in groups.items():
            if len(ks) != 4:
                continue
            st = part.status[ks]
            same_decided = st[0] >= 0 and np.all(st == st[0])
            empty = np.all(counts[ks] == 0)
            if same_decided or empty:
                merge.append((parent, ks))
        if not merge:
            return part
        drop = np.zeros(part.n_leaves, dtype=bool)
        nix, niy, nd, ncnt = [], [], [], []
        for (d, i, j), ks in merge:
            drop[ks] = True
            nix.append(i), niy.append(j), nd.append(d), ncnt.append(int(counts[ks].sum()))
        keep = ~drop
        part = _with_status(Partition(part.root, np.concatenate([part.ix[keep], np.array(nix, dtype=np.int64)]),
                                      np.concatenate([part.iy[keep], np.array(niy, dtype=np.int64)]),
                                      np.concatenate([part.depth[keep], np.array(nd, dtype=np.int64)]),
                                      np.zeros(int(keep.sum()) + len(nix), dtype=np.int8)), fam)
        counts = np.concatenate([counts[keep], np.array(ncnt, dtype=counts.dtype)])
