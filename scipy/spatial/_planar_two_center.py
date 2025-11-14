import numpy as np
from scipy.spatial import ConvexHull


def find_farthest_pair(S):
    # The preprocessing step for theorem 1 in Cho et al. (2024)
    # includes finding the farthest pair (a, b) of the input planar
    # points (S), and should require O(n log n) time according to their
    # analysis.

    # the convex hull calculation is well known to be O(n log n):
    hull = ConvexHull(S)
    # indexing the points to retrieve the hull is O(n) at worst:
    hull_coordinates = S[hull.vertices]

    # the diameter of the convex hull may be calculated in O(n) time
    # using https://en.wikipedia.org/wiki/Rotating_calipers from
    # Shamos' 1978 PhD dissertation, and is equivalent to the farthest
    # pair (a, b) in S
