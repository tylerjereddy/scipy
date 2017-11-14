"""

Polygon Surface Area Code

.. versionadded:: 1.0.0

"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.spatial.distance import pdist
from . import _surface_area
from six.moves import xrange

#
# Copyright (C)  James Nichols and Tyler Reddy
# Contributions from Tyler Reddy are owned by LANL
# and respect the scipy distribution license
#
# Distributed under the same BSD license as Scipy.

def convert_cartesian_array_to_spherical_array(coord_array,angle_measure='radians'):
    '''Take shape (N,3) cartesian coord_array and return an array of the same shape in spherical polar form (r, theta, phi). Based on StackOverflow response: http://stackoverflow.com/a/4116899
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    spherical_coord_array = np.zeros(coord_array.shape)
    xy = coord_array[...,0]**2 + coord_array[...,1]**2
    spherical_coord_array[...,0] = np.sqrt(xy + coord_array[...,2]**2)
    spherical_coord_array[...,1] = np.arctan2(coord_array[...,1], coord_array[...,0])
    spherical_coord_array[...,2] = np.arccos(coord_array[...,2] / spherical_coord_array[...,0])
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = np.degrees(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = np.degrees(spherical_coord_array[...,2])
    return spherical_coord_array

def convert_spherical_array_to_cartesian_array(spherical_coord_array,angle_measure='radians'):
    '''Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of the same shape in cartesian coordinate form (x,y,z). Based on the equations provided at: http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    cartesian_coord_array = np.zeros(spherical_coord_array.shape)
    #convert to radians if degrees are used in input (prior to Cartesian conversion process)
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = np.deg2rad(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = np.deg2rad(spherical_coord_array[...,2])
    #now the conversion to Cartesian coords
    cartesian_coord_array[...,0] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,1] = spherical_coord_array[...,0] * np.sin(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,2] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,2])
    return cartesian_coord_array

def _vertex_index_strider(index, num_vertices):
    # handle the wrapping / iteration over
    # polygon vertices in either CW or CCW
    # sort order
    forward_index = index + 1
    backward_index = index - 1
    if forward_index >= num_vertices:
        forward_index = 0
    return forward_index, backward_index

def _planar_polygon_area(vertices):
    num_vertices = vertices.shape[0]
    area_sum = 0
    for i in xrange(num_vertices):
        forward_index, backward_index = _vertex_index_strider(i, num_vertices)
        delta_x = (vertices[forward_index][0] -
                   vertices[backward_index][0])
        area_sum += delta_x * vertices[i][1]
    area = -0.5 * area_sum
    return area

def _slerp(start_coord,
           end_coord,
           n_pts):
    # spherical linear interpolation between points
    # on great circle arc
    # see: https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    # NOTE: could we use scipy.interpolate.RectSphereBivariateSpline instead?
    omega = np.arccos(np.dot(start_coord, end_coord))
    t_values = np.linspace(0, 1, n_pts)
    new_pts = []
    for t in t_values:
        new_pt = (((np.sin((1 - t) * omega) / np.sin(omega)) * start_coord) +
                  ((np.sin(t * omega) / np.sin(omega)) * end_coord))
        new_pts.append(new_pt)
    return np.array(new_pts)

def _spherical_polygon_area(vertices, radius, discretizations):
    num_vertices = vertices.shape[0]
    area_sum = 0

    for i in xrange(num_vertices):
        new_pts = _slerp(vertices[i], vertices[i-1], discretizations)

        lambda_range = np.arctan2(new_pts[...,1], new_pts[...,0])
        phi_range = np.arcsin((new_pts[...,2]))
        area_element = 0
        for j in xrange(discretizations - 1):
            delta_lambda = (lambda_range[j+1] -
                            lambda_range[j])
            second_term = 2 + np.sin(phi_range[j]) + np.sin(phi_range[j+1])
            area_element += (delta_lambda * second_term * (radius ** 2) * 0.5)
        area_sum += area_element
    area = area_sum
    return area

def poly_area(vertices, radius=None, threshold=1e-21,
              cython=None, discretizations=500):
    # calculate the surface area of a planar or spherical polygon
    # crude pure Python implementation for handling a single
    # polygon at a time
    # based on JPL Publication 07-3 by Chamberlain and Duquette (2007)
    # for planar polygons we currently still require x,y,z coords
    # can just set i.e., z = 0 for all vertices
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

        # normalize vertices to unit sphere
        vertices = convert_cartesian_array_to_spherical_array(vertices)
        vertices[...,0] = 1.
        vertices = convert_spherical_array_to_cartesian_array(vertices)

        if cython is None:
            area = _spherical_polygon_area(vertices, radius, discretizations)
        else: # cython code for spherical polygon SA
            area = _spherical_polygon_area(vertices, radius, discretizations)
            #area = _surface_area.spherical_polygon_area(vertices, radius,
                                                        #discretizations)
    else: # planar polygon
        if cython is None:
            area = _planar_polygon_area(vertices)
        else: # cython code for planar polygon SA
            area = _surface_area.planar_polygon_area(vertices)
    
    return abs(area)






