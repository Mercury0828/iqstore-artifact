"""IQSTORE ingest of one position: certified partition + stored bytes.

`build_partition` stops at the first certified partition, which is not byte-optimal: a coarser certified partition puts
more shots in the band (each band shot stores a leaf id instead of one side bit), so b-bar can even grow with tau (DEV:
F-boot, tau = 1e-2 above tau = 1e-3 on GHZ12_a). Ingest-time search (the analogue of B4q's grid-step search): build the
partition for the certificate budgets floor(f * tau * N), f in FACTORS = 2^-k, k = 0..12 (each certified partition at f * tau <= tau is
certified at tau), serialize each candidate completely, keep the smallest (ties: first in FACTORS order). The search
stops after two consecutive candidates that are not smaller than the best so far, after one candidate more than 5 %
larger than the best, or at budget 0 (smaller budgets make construction much slower and, past the optimum, larger). Search time is construction time.
"""
from __future__ import annotations

import numpy as np

from .family import Family
from .frame import Frame
from .partition import build_partition
from .rep import PositionRep, encode_position
from .storage import iqstore_blobs

FACTORS = tuple(2.0 ** -k for k in range(13))  # 1, 1/2, ..., 1/4096; the loop stops at budget 0


def ingest_position(frame: Frame, fam: Family, x: np.ndarray, y: np.ndarray, raw: np.ndarray, tau: float, d0: int,
                    max_depth: int, split_frac: float, factors=FACTORS) -> tuple[PositionRep, list[bytes], dict]:
    best = None
    worse = 0
    tried = []
    for f in factors:
        part, exc, cert, _ = build_partition(x, y, fam, f * tau, d0=d0, max_depth=max_depth, split_frac=split_frac)
        if not cert.ok:
            raise AssertionError("partition not certified")
        rep = encode_position(frame, fam, part, x, y, raw, exc)
        blobs = iqstore_blobs(rep)
        size = sum(len(b) + 4 for b in blobs)
        tried.append((f, size))
        if best is None or size < best[0]:
            best = (size, f, rep, blobs, part, exc)
            worse = 0
        else:
            worse += 1
            if worse >= 2 or size > 1.05 * best[0]:
                break
        if int(np.floor(f * tau * x.size)) == 0:
            break  # budget 0 reached: smaller factors give the same partition
    size, f, rep, blobs, part, exc = best
    return rep, blobs, {"factor": f, "bytes": size, "tried": tried, "part": part, "exc": exc}
