import math

from scipy.spatial import planar_k_center, find_farthest_pair
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
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
    ),
    # when the input points fall on two non-overlapping input circles of
    # different radii, planar_k_center should return two disks that are centered
    # at the centers of each of those two circles, but with the radius of each
    # disk set to the larger of the radii so that all points get covered while
    # simultaneously satisfying the requirement for congruent disks

    (
    # input points lying on "circle 1," centered at the origin
    # with radius 2:
    [[0, 2],
    [2, 0],
    [0, -2],
    [-2, 0],
    # input points lying on "circle 2," centered at +100 x coordinate
    # and with DOUBLE radius of circle 1 (radius of 4)
    [100, 4],
    [104, 0],
    [100, -4],
    [96, 0],
    ],
    [0, 0], # one disk is centered at the origin
    [100, 0], # the other disk is centered +100 from the origin
    4, # the largest of the radii is preserved for the returned disks
    ),
])
def test_simple_cases(points, exp_e0, exp_e1, exp_r):
    # verify cases where the expected outcome can be
    # deduced using simple mathematical intuition
    result = planar_k_center(points=points, k=2)
    assert_allclose(result.e0, exp_e0)
    assert_allclose(result.e1, exp_e1)
    assert_allclose(result.r, exp_r)


@pytest.mark.parametrize("points, exp_max_dist, exp_max_pair", [
    # the farthest distance in a "unit square" is the
    # diagonal (root 2); technically there are two such
    # possible diagonals; we check that one of them is detected:
    ([[0, 0], [1, 0], [1, 1], [0, 1]],
     math.sqrt(2),
     [[0, 0], [1, 1]]),
])
def test_find_farthest_pair(points, exp_max_dist, exp_max_pair):
    actual_max_dist, actual_max_pair = find_farthest_pair(points)
    assert_allclose(actual_max_dist, exp_max_dist)
    assert_array_equal(actual_max_pair, exp_max_pair)
