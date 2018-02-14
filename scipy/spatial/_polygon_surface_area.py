"""

Polygon Surface Area Code

.. versionadded:: 1.0.0

"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.spatial.distance import pdist, cdist
from . import _surface_area
from six.moves import xrange
import math

#
# Copyright (C)  James Nichols and Tyler Reddy
#
# Distributed under the same BSD license as Scipy.

def poly_area(vertices, radius=None, threshold=1e-21,
              discretizations=500,
              n_rot=50):
    # calculate the surface area of a planar or spherical polygon
    # crude pure Python implementation for handling a single
    # polygon at a time
    # based on JPL Publication 07-3 by Chamberlain and Duquette (2007)
    # for planar polygons we currently still require x,y,z coords
    # can just set i.e., z = 0 for all vertices
    num_vertices = vertices.shape[0]
    # require that a planar or spherical triangle is
    # the smallest possible input polygon
    if num_vertices < 3:
        err_str = "An input polygon must have at least 3 vertices."
        raise ValueError(err_str)

    min_vertex_dist = pdist(vertices).min()
    if min_vertex_dist < threshold:
        err_str = '''Duplicate vertices detected based on minimum
                     distance {min_vertex_dist} and threshold value
                     {threshold}.'''.format(min_vertex_dist=min_vertex_dist,
                                            threshold=threshold)
        raise ValueError(err_str)


    if radius is not None: # spherical polygons
        if radius <= threshold:
            err_str = 'radius must be > {threshold}'.format(threshold=threshold)
            raise ValueError(err_str)

        # if any two *consecutive* vertices in the spherical polygon
        # are antipodes, there's an infinite set of
        # geodesics connecting them, and we cannot
        # possibly guess the appropriate surface area
        dist_matrix = cdist(vertices, vertices)
        # TODO: use a threshold for the floating
        # point dist comparison here
        matches = np.where(dist_matrix == (2. * radius))
        # can't handle consecutive matches that are antipodes
        if np.any(np.abs(matches[0] - matches[1]) == 1):
            raise ValueError("Consecutive antipodal vertices are ambiguous.")

        # a great circle can only occur if the origin lies in a plane
        # with all vertices of the input spherical polygon
        # the input spherical polygon must have at least three vertices
        # to begin with (i.e., spherical triangle as smallest possible
        # spherical polygon) so we can safely assume (and check above)
        # three input vertices plus the origin for our test here

        # points lie in a common plane if the determinant below
        # is zero (informally, see:
        # https://math.stackexchange.com/a/684580/480070 
        # and http://mathworld.wolfram.com/Plane.html eq. 18)

        # have to iterate through the vertices in chunks
        # to preserve the three-point form of the plane-check
        # determinant
        current_vert = 0
        plane_failures = 0
        one_pads = np.ones((1, 4))
        while current_vert <= (num_vertices - 3):
            candidate_plane = np.concatenate((np.zeros((1,3)),
                                              vertices[current_vert:current_vert + 3]))

            candidate_plane = np.concatenate((candidate_plane.T,
                                              one_pads))

            if np.linalg.det(candidate_plane) != 0:
                # if any of the vertices aren't on the plane with
                # the origin we can safely break out
                plane_failures += 1
                break

            current_vert += 1

        if plane_failures == 0:
            # we have a great circle so
            # return hemisphere area
            return 2. * np.pi * (radius ** 2) 

        # normalize vertices to unit sphere
        vertices = _surface_area.convert_cartesian_array_to_spherical_array(vertices)
        vertices[...,0] = 1.
        vertices = _surface_area.convert_spherical_array_to_cartesian_array(vertices)

        # try to handle spherical polygons that contain
        # a pole by rotating them; failing that,
        # return an area of np.nan

        # rotate around y axis to move away
        # from the N/ S poles
        rot_axis = np.array([0,1,0])
        while _surface_area.pole_in_polygon(vertices) == 1:
            n_rot -= 1
            rot_angle = np.random.random_sample() * (np.pi / 6.)
            if n_rot == 0:
                return np.nan
            else:
               # use Rodrigues' rotation formula
               for index in range(num_vertices):
                  row = vertices[index]
                  vertices[index] = (row * math.cos(rot_angle) +
                               np.cross(rot_axis, row) * math.sin(rot_angle) +
                               rot_axis * (np.dot(row, rot_axis)) * (1 - math.cos(rot_angle)))

        area = _surface_area._spherical_polygon_area(vertices,
                                                    radius,
                                                    discretizations)
    else: # planar polygon
        area = _surface_area.planar_polygon_area(vertices)
    
    return abs(area)
