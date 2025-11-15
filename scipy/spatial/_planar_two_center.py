import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean


def _get_index(size, idx):
    if idx == size:
        return 0
    else:
        return idx


def shoelace_double(x, y, z):
    area_db = (y[0] - x[0]) * (z[1] - x[1]) - (z[0] - x[0]) * (y[1] - x[1])
    return area_db


def find_farthest_pair(S):
    # The preprocessing step for theorem 1 in Cho et al. (2024)
    # includes finding the farthest pair (a, b) of the input planar
    # points (S), and should require O(n log n) time according to their
    # analysis.
    S = np.asarray(S)

    # the convex hull calculation is well known to be O(n log n):
    hull = ConvexHull(S)
    # indexing the points to retrieve the hull is O(n) at worst:
    hull_coordinates = S[hull.vertices]

    # the diameter of the convex hull may be calculated in O(n) time
    # using https://en.wikipedia.org/wiki/Rotating_calipers from
    # Shamos' 1978 PhD dissertation, and is equivalent to the farthest
    # pair (a, b) in S
    antipodal_pairs = []
    s = hull.vertices.size
    n = hull.vertices.size - 1
    i = hull.vertices.size - 1
    j = 0
    while (shoelace_double(hull_coordinates[_get_index(s, i)],
                          hull_coordinates[_get_index(s, i + 1)],
                          hull_coordinates[_get_index(s, j + 1)]) > 
           shoelace_double(hull_coordinates[_get_index(s, i)],
                           hull_coordinates[_get_index(s, i + 1)],
                           hull_coordinates[_get_index(s, j)])):
        j += 1
        j = _get_index(s, j)
    j0 = j
    while (i != j0):
        i += 1
        i = _get_index(s, i)
        antipodal_pairs.append([i, j])
        while (shoelace_double(hull_coordinates[_get_index(s, i)],
                              hull_coordinates[_get_index(s, i + 1)],
                              hull_coordinates[_get_index(s, j + 1)]) > 
               shoelace_double(hull_coordinates[_get_index(s, i)],
                               hull_coordinates[_get_index(s, i + 1)],
                               hull_coordinates[_get_index(s, j)])):
            j += 1
            j = _get_index(s, j)
            if (i, j) != (j0, 1):
                antipodal_pairs.append([i, j])
        if (shoelace_double(hull_coordinates[_get_index(s, i)],
                              hull_coordinates[_get_index(s, i + 1)],
                              hull_coordinates[_get_index(s, j + 1)]) > 
            shoelace_double(hull_coordinates[_get_index(s, i)],
                               hull_coordinates[_get_index(s, i + 1)],
                               hull_coordinates[_get_index(s, j)])):
            if (i, j) != (j0, n):
                antipodal_pairs.append([i, j + 1])
    max_dist = 0
    for pair in antipodal_pairs:
        x = hull_coordinates[pair[0]]
        y = hull_coordinates[pair[1]]
        dist = euclidean(x, y)
        if dist > max_dist:
            max_dist = dist
            max_pair = [x, y]
    return max_dist, max_pair


def planar_k_center(points, k):
    # algorithm preprocessing starts with O(n log n)
    # identification of the max distance points and their
    # midpoint
    max_dist, max_pair = find_farthest_pair(points)
    a_b_midpoint = (max_pair[0] + max_pair[1]) / 2
