"""Fast deterministic regression tests (run: python -m pytest tests -q)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from iqstore.family import Discriminator, Family, point_decided  # noqa: E402
from iqstore.partition import certify  # noqa: E402
from iqstore.rep import LS_BOTH, LS_ONLY0, rect_label_sets  # noqa: E402


def test_wraparound_trap():
    # w = (-1, 0) must NOT be certified for theta in [-0.1, 0.1] by a test without the 2k*pi shift
    fam = Family(-0.1, 0.1, -0.5, 0.5)
    assert point_decided(np.array([-1.0]), np.array([0.0]), fam)[0] == 0
    fam2 = Family(np.pi - 0.1, np.pi + 0.1, -0.5, 0.5)  # wraps around pi
    assert point_decided(np.array([-1.0]), np.array([0.0]), fam2)[0] == 1


def test_half_open_cell_on_line():
    # cell [0,1)x[0,1), line x = 1 (theta 0, beta 1): no cell point has x > 1 -> real label set {0}; the guarded
    # implementation may report {0,1}, never {1}
    ls = rect_label_sets(0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert ls & LS_ONLY0


def test_certifier_matches_guarded_label_sets():
    # certify must not claim zero uncertainty where the query path reports {0,1}
    res = certify([0.0], [0.0], [1.0], [1.0], [1], Family(-0.1, 0.0, 1.0, 1.1), budget=0)
    ls = rect_label_sets(0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert not (res.ok and ls == LS_BOTH)


def test_strict_convention():
    h = Discriminator(0.0, 0.5)
    assert not h.label(np.array([0.5]), np.array([0.0]))[0]
