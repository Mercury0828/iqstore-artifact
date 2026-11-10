"""A7 — family-agnostic certified partition (prereg/M2.md v4 §3, [F4]).

Leaf geometry chosen WITHOUT the family: a density-adaptive quadtree on the same dyadic root as IQSTORE that splits every
leaf holding more than c shots, c over {ceil(N 2^-k) : k = 1..16}, down to depth A7_MAX_DEPTH = 32 (a leaf at that depth is
not split further; its shots are handled by the exception safety net if a certificate needs it). Leaf status by the same exact corner test against the declared family; NO family-driven merge (the
geometry stays family-agnostic); band leaves stored as in IQSTORE (same serializer and coder; the tree in its generic
form, one split bit per node, because an unpruned tree can hold decided internal nodes). Each
candidate is certified with the same continuous-family certifier (`certify_partition`); the smallest certified complete
serialization is kept (ties: smaller k). As in IQSTORE, exceptions are the certificate's safety net: while a candidate
fails, the shots of the certifier's hot band leaves (crossed by the witness, or undecided in the stuck sub-box) move to
the exception list (stored exactly). All 16 candidates are evaluated (sizes are not unimodal in k: exceptions
fall and leaves grow as c shrinks). If no candidate certifies, the position falls back to exact storage (B0 bytes
of that column), reported.
"""
from __future__ import annotations

import math

import numpy as np

from iqstore.family import Family
from iqstore.frame import Frame
from iqstore.partition import _counts, _with_status, certify_partition
from iqstore.rep import COORD_MAX, Partition, dyadic_root, encode_position
from iqstore.storage import iqstore_blobs

A7_MAX_DEPTH = 32  # the deepest exactly representable and exactly locatable leaf (rep.LOCATE_MAX <= D_MAX); fixed
#                    before any A7 result; not inherited from IQSTORE's DEV grid
MAX_EXC_ROUNDS = 500


def density_tree(x: np.ndarray, y: np.ndarray, c: int, root) -> Partition:
    ix = np.zeros(1, dtype=np.int64)
    iy = np.zeros(1, dtype=np.int64)
    dp = np.zeros(1, dtype=np.int64)
    while True:
        part = Partition(root, ix, iy, dp, np.zeros(ix.size, dtype=np.int8))
        cnt = np.bincount(part.locate(x, y), minlength=ix.size)
        split = (cnt > c) & (dp < A7_MAX_DEPTH)
        if not split.any():
            return part
        keep = ~split
        nix, niy, ndp = [ix[keep]], [iy[keep]], [dp[keep]]
        for a in (0, 1):
            for b in (0, 1):
                nix.append(2 * ix[split] + a)
                niy.append(2 * iy[split] + b)
                ndp.append(dp[split] + 1)
        ix, iy, dp = np.concatenate(nix), np.concatenate(niy), np.concatenate(ndp)


def a7_position(frame: Frame, fam: Family, x: np.ndarray, y: np.ndarray, raw: np.ndarray, tau: float):
    """Returns (rep or None, blobs, info). rep is None on exact fallback (blobs then hold the raw column)."""
    N = x.size
    budget = int(np.floor(tau * N))
    ok = (np.abs(x) <= COORD_MAX) & (np.abs(y) <= COORD_MAX)
    if not ok.all():
        raise ValueError("A7: coordinate outside the dyadic-root domain")
    root = dyadic_root(x, y)
    best = None
    tried = []
    for k in range(1, 17):
        c = math.ceil(N * 2.0 ** -k)
        part = _with_status(density_tree(x, y, c, root), fam)
        leaf = part.locate(x, y)
        exc = np.zeros(N, dtype=bool)
        for _ in range(MAX_EXC_ROUNDS):
            res, idx = certify_partition(part, _counts(part, leaf, ~exc), fam, budget)
            if res.ok:
                break
            exc |= np.isin(leaf, idx[res.hot])  # shots of the hot band leaves become exceptions (exact points)
        if not res.ok:
            tried.append((k, None))
            continue
        rep = encode_position(frame, fam, part, x, y, raw, exc)
        blobs = iqstore_blobs(rep, generic_tree=True)
        size = sum(len(b) + 4 for b in blobs)
        tried.append((k, size))
        if best is None or size < best[0]:
            best = (size, k, rep, blobs)
    if best is None:
        import struct
        from iqstore.storage import DTYPE_CODE, zbest
        raw = np.ascontiguousarray(raw)
        fdt = np.float32 if raw.dtype == np.complex64 else np.float64
        hdr = struct.pack("<B4d4dqB", 0, *frame.to_array(), *fam.to_array(), N, DTYPE_CODE[raw.dtype])  # mode 0 = exact
        blobs = [hdr, zbest(raw.view(fdt))]
        return None, blobs, {"mode": "exact-fallback", "tried": tried}
    size, k, rep, blobs = best
    return rep, blobs, {"mode": "tree", "k": k, "bytes": size, "tried": tried}
