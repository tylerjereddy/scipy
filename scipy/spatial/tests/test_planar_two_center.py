from scipy.spatial import planar_k_center

import numpy as np
from numpy.testing import assert_allclose
import pytest



@pytest.mark.parametrize("points, exp_e0, exp_e1, exp_r", [
    # for two points separated by a distance `d`, the
    # congruent disks of minimal size must have the same
    # r = d/2 and matching center position (they are
    # perfectly superposed)
    ([[0, 0], [0, 4]],
     [0, 2],
     [0, 2],
     2,
    ),
])
def test_simple_cases(points, exp_e0, exp_e1, exp_r):
    # verify cases where the expected outcome can be
    # deduced using simple mathematical intuition
    result = planar_k_center(points=points, k=2)
    assert_allclose(result.e0, exp_e0)
    assert_allclose(result.e1, exp_e1)
    assert_allclose(result.r, exp_r)
