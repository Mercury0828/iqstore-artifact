"""Per-position affine frame T_j fixed at ingest.

From the calibration pair of the ingest day (all-|0> and all-|1> preparation) take the class means mu0, mu1 (complex).
T_j translates the midpoint to the origin, rotates mu1 - mu0 onto the positive first axis and scales by |mu1 - mu0| / 2,
so that mu0 -> (-1, 0) and mu1 -> (+1, 0). Four stored numbers: centre (2), angle, scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    cx: float  # centre, real part (raw units)
    cy: float  # centre, imaginary part (raw units)
    phi: float  # angle of mu1 - mu0 in the raw IQ plane (rad)
    scale: float  # |mu1 - mu0| / 2 (raw units)

    def apply(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map raw complex IQ points to frame coordinates (x, y), float64."""
        w = (np.asarray(z, dtype=np.complex128) - complex(self.cx, self.cy)) * np.exp(-1j * self.phi) / self.scale
        return w.real.copy(), w.imag.copy()

    def to_array(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.phi, self.scale], dtype=np.float64)

    @staticmethod
    def from_array(a: np.ndarray) -> "Frame":
        return Frame(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def fit_frame(cal0: np.ndarray, cal1: np.ndarray) -> Frame:
    """Frame from calibration IQ points of one position: cal0 prepared |0>, cal1 prepared |1> (complex arrays)."""
    mu0 = complex(np.mean(np.asarray(cal0, dtype=np.complex128)))
    mu1 = complex(np.mean(np.asarray(cal1, dtype=np.complex128)))
    d = mu1 - mu0
    if abs(d) == 0.0:
        raise ValueError("class means coincide; frame undefined")
    c = 0.5 * (mu0 + mu1)
    return Frame(c.real, c.imag, float(np.angle(d)), abs(d) / 2.0)
