"""Phase-0 minimal mitigation setup (the full versioned-view layer, is in eval/ and iqstore/e2e.py).

Calibration datasets are stored RAW, so for any discriminator h_j the assignment matrix
A_j[i, k] = P(label i | prepared k) is computed exactly from the stored calibration shots classified by h_j.
The fixed mitigation operator of a view over positions S is M = kron_{j in S} A_j^{-1} (tensor-product order: the i-th
position of S is bit i of the outcome index, so the kron is taken in reverse position order).
A mitigated linear functional <f, M p> with p the empirical label distribution equals (1/N) sum_s (M^T f)(l_s), i.e. a
separable aggregate with g = M^T f. Because M depends on h only through exactly stored calibration rows, the interval of
T1 applies to it unchanged for a FIXED h (the operator is exact, only the target shots' labels are uncertain).
"""
from __future__ import annotations

import numpy as np

from .family import Discriminator
from .frame import Frame


def assignment_matrix(cal0_raw: np.ndarray, cal1_raw: np.ndarray, frame: Frame, h: Discriminator) -> np.ndarray:
    x0, y0 = frame.apply(cal0_raw)
    x1, y1 = frame.apply(cal1_raw)
    p10 = float(np.mean(h.label(x0, y0)))  # P(1 | prepared 0)
    p11 = float(np.mean(h.label(x1, y1)))  # P(1 | prepared 1)
    return np.array([[1.0 - p10, 1.0 - p11], [p10, p11]])


def mitigation_g(A_list: list[np.ndarray], f: np.ndarray) -> np.ndarray:
    """g = M^T f with M = kron of inverses; A_list[i] belongs to position i of S (bit i of the index)."""
    M = np.array([[1.0]])
    for A in A_list:  # position i becomes bit i: kron(A_i^{-1}, M_previous) puts it on the high side -> reverse
        M = np.kron(np.linalg.inv(A), M)
    return M.T @ np.asarray(f, dtype=np.float64)
