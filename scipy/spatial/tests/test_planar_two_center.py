import math

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
    # when the input points are on a perfect circle centered at the
    # origin (in this case, the unit circle), it is clear that both
    # disks must coincide exactly with that circle of points to be
    # both congruent and to minimize thier radii
    ([[0, 1],
     [math.sqrt(2)/2, math.sqrt(2)/2],
     [1, 0],
     [math.sqrt(2)/2, -math.sqrt(2)/2],
     [0, -1],
     [-math.sqrt(2)/2, -math.sqrt(2)/2],
     [-1, 0],
     [-math.sqrt(2)/2, math.sqrt(2)/2]],
     [0, 0], # first center
     [0, 0], # second center
     1, # radius
    ),
    # when the input points fall on two non-overlapping circles of the same radius
    # then the resulting planar_k_center disks should correspond to the centers/radii
    # of those input circles
    (
    [
    # input points lying on "circle 1," centered at the origin
    # with radius 2:
    [0, 2],
    [2, 0],
    [0, -2],
    [-2, 0],
    # input points lying on "circle 2," centered at +100 x coordinate
    # but otherwise the same (radius of 2):
    [100, 2],
    [102, 0],
    [100, -2],
    [98, 0],
    ],
    [0, 0], # one disk is centered at the origin
    [100, 0], # the other disk is centered +100 from the origin
    2,
    )
])
def test_simple_cases(points, exp_e0, exp_e1, exp_r):
    # verify cases where the expected outcome can be
    # deduced using simple mathematical intuition
    result = planar_k_center(points=points, k=2)
    assert_allclose(result.e0, exp_e0)
    assert_allclose(result.e1, exp_e1)
    assert_allclose(result.r, exp_r)
