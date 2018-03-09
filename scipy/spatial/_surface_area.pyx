import numpy as np
from scipy.spatial.distance import pdist, cdist
cimport numpy as np
cimport cython
from libc.math cimport (sin, acos, atan2,
                        cos, M_PI, abs)
from libc.stdlib cimport RAND_MAX, rand

cdef int vertex_index_strider(int index, int num_vertices):
    cdef int forward_index
    forward_index = index + 1
    if forward_index > (num_vertices - 1):
        forward_index = 0
    return forward_index

@cython.boundscheck(False)
@cython.wraparound(False)
def planar_polygon_area(double[:,::1] vertices):
    cdef int N = vertices.shape[0]
    cdef int i, forward_index, backward_index
    cdef double area = 0
    cdef double delta_x

    for i in range(N):
        forward_index = vertex_index_strider(i, N)
        backward_index = i - 1
        delta_x = (vertices[forward_index, 0] -
                   vertices[backward_index, 0])
        area += delta_x * vertices[i, 1]
    area *= 0.5
    return area

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _slerp(double[:] start_coord,
            double[:] end_coord,
            int n_pts,
            double[::1] t_values):
    # spherical linear interpolation between points
    # on great circle arc
    # see: https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    # NOTE: could we use scipy.interpolate.RectSphereBivariateSpline instead?
    cdef:
        double omega = acos(np.dot(start_coord, end_coord))
        double sin_omega = sin(omega)
        double[:,:] new_pts = np.empty((n_pts, 3), dtype=np.float64)
        int i, j
        double factors[2]

    for i in xrange(n_pts):
        factors[0] = sin((1 - t_values[i]) * omega) / sin_omega
        factors[1] = sin(t_values[i] * omega) / sin_omega
        for j in range(3):
            new_pts[i,j] = ((factors[0] * start_coord[j]) +
                            (factors[1] * end_coord[j]))
    return new_pts

@cython.cdivision(True)
@cython.boundscheck(False)
cdef calc_heading(double[:] lambda_range,
                  double[:] phi_range,
                  int i,
                  int next_index):

    cdef double phi_1 = phi_range[i]
    cdef double phi_2 = phi_range[next_index]
    cdef double delta_lambda = lambda_range[i] - lambda_range[next_index]
    cdef double result = atan2(sin(delta_lambda) * cos(phi_2),
                               cos(phi_1) * sin(phi_2) -
                               sin(phi_1) * cos(phi_2) * cos(delta_lambda))
    cdef double course_angle = result % (2 * M_PI) * 180.0 / M_PI
    return course_angle

def pole_in_polygon(double [:,:] vertices, double[::1] t_values):
    # determine if the North or South Pole is contained
    # within the spherical polygon defined by vertices
    # the implementation is based on discussion
    # here: https://blog.element84.com/determining-if-a-spherical-polygon-contains-a-pole.html
    # and the course calculation component of the aviation
    # formulary here: http://www.edwilliams.org/avform.htm#Crs

    #TODO: handle case where initial point is within
    # i.e., expected machine precision of a pole

    #TODO: handle case where both the N & S poles
    # are contained within a given polygon

    cdef double[:] lambda_range = np.arctan2(vertices[...,1], vertices[...,0])
    cdef double[:] phi_range = np.arcsin((vertices[...,2]))
    cdef double[:] new_lambda_range, new_phi_range
    cdef int i
    cdef int N = vertices.shape[0]
    cdef double[:,:] new_pts
    cdef double net_angle, course_angle
    cdef int next_index
    cdef int counter = 0
    cdef int n_pts = 900
    cdef double[:] course_angles = np.empty((2 * N + 1), dtype=np.float64)

    for i in range(N):
        if i == (N - 1):
            next_index = 0
        else:
            next_index = i + 1

        # calculate heading when leaving point 1
        # the "departure" heading
        course_angle = calc_heading(lambda_range,
                                    phi_range,
                                    i,
                                    next_index)

        course_angles[counter] = course_angle

        # calcualte heading when arriving at point 2
        # strategy: discretize arc path between
        # points 1 and 2 and take the penultimate
        # point for bearing calculation
        new_pts = _slerp(vertices[i], vertices[next_index],
                         n_pts=n_pts,
                         t_values=t_values)
        new_lambda_range = np.arctan2(new_pts[...,1], new_pts[...,0])
        new_phi_range = np.arcsin((new_pts[...,2]))

        # the "arrival" heading is estimated between
        # penultimate (-2) and final (-1) index
        # points
        course_angle = calc_heading(new_lambda_range,
                                    new_phi_range,
                                    -2,
                                    -1)
        course_angles[counter + 1] = course_angle
        counter += 2

    course_angles[counter] = course_angles[0]
    angles = np.array(course_angles)
    # delta values should be capped at 180 degrees
    # and follow a CW or CCW directionality
    deltas = angles[1:] - angles[:-1]
    deltas[deltas > 180] -= 360
    deltas[deltas < -180] += 360
    net_angle = np.sum(deltas)

    if np.allclose(net_angle, 0.0):
        # there's a pole in the polygon
        return 1
    elif np.allclose(abs(net_angle), 360.0):
        # no single pole in the polygon
        # a normal CW or CCW-sorted
        # polygon
        return 0
    else:
        # something went wrong
        return -1

@cython.boundscheck(False)
cdef _spherical_polygon_area(double[:,:] vertices,
                             double radius,
                             int discretizations,
                             double[::1] t_values):
    cdef:
        int num_vertices = vertices.shape[0]
        double area_sum = 0
        double second_term, delta_lambda
        int i, j
        double[:,:] new_pts

    for i in xrange(num_vertices):
        new_pts = _slerp(vertices[i], vertices[i-1], discretizations,
                         t_values)
        for j in xrange(discretizations - 1):
            delta_lambda = (atan2(new_pts[j + 1, 1], new_pts[j + 1, 0]) -
                            atan2(new_pts[j, 1], new_pts[j, 0]))

            # at the + / - pi transition point
            # of the unit circle
            # add or subtract 2 * pi to the delta
            # based on original sign
            if delta_lambda > M_PI:
                delta_lambda -= 2 * M_PI
            elif delta_lambda < (-M_PI):
                delta_lambda += 2 * M_PI

            second_term = 2 + new_pts[j, 2] + new_pts[j + 1, 2]
            area_sum += (delta_lambda * second_term * 0.5)
    return area_sum * radius * radius

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef normalize_to_unit_sphere(double[:,:] coord_array,
                              double radius):
    '''Normalize coordinates to the unit sphere.'''
    cdef:
        int N = coord_array.shape[0]
        int i
        double spherical_vals[2]
        double[:,:] cartesian_coord_array = np.zeros((N, 3), dtype=np.float64)

    for i in xrange(N):
        spherical_vals[0] = atan2(coord_array[i,1], coord_array[i,0])
        spherical_vals[1] = acos(coord_array[i,2] / radius)
        cartesian_coord_array[i,0] = cos(spherical_vals[0]) * sin(spherical_vals[1])
        cartesian_coord_array[i,1] = sin(spherical_vals[0]) * sin(spherical_vals[1])
        cartesian_coord_array[i,2] = cos(spherical_vals[1])
    return cartesian_coord_array

cdef poly_area(double[:,:] vertices,
               double[::1] t_values,
               radius=None,
               double threshold=1e-21,
               int discretizations=500,
               int n_rot=50):
    # calculate the surface area of a planar or spherical polygon
    # crude pure Python implementation for handling a single
    # polygon at a time
    # based on JPL Publication 07-3 by Chamberlain and Duquette (2007)
    # for planar polygons we currently still require x,y,z coords
    # can just set i.e., z = 0 for all vertices
    cdef int num_vertices = vertices.shape[0]
    cdef double[:,:] one_pads = np.ones((1, 4), dtype=np.float64)
    cdef double[:,:] candidate_plane
    cdef int current_vert, plane_failures, k, index
    cdef double[:] rot_axis = np.array([0,1,0], dtype=np.float64)
    cdef double rot_angle, area, dot_prod, min_vertex_dist
    cdef double angle_factor = M_PI / 6.
    cdef double[:] cross_prod, row
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
            return 2. * M_PI * (radius ** 2) 

        # normalize vertices to unit sphere
        vertices = normalize_to_unit_sphere(vertices, radius)

        # try to handle spherical polygons that contain
        # a pole by rotating them; failing that,
        # return an area of np.nan

        # rotate around y axis to move away
        # from the N/ S poles
        while pole_in_polygon(vertices, t_values) == 1:
            n_rot -= 1
            rot_angle = rand() / <float> RAND_MAX * angle_factor
            if n_rot == 0:
                return np.nan
            else:
               # use Rodrigues' rotation formula
               for index in range(num_vertices):
                  cross_prod = np.cross(rot_axis, vertices[index])
                  dot_prod = np.dot(vertices[index], rot_axis)
                  for k in range(3):
                      vertices[index, k] = (vertices[index,k] * cos(rot_angle) +
                                   cross_prod[k] * sin(rot_angle) +
                                   rot_axis[k] * dot_prod * (1 - cos(rot_angle)))

        area = _spherical_polygon_area(vertices,
                                       radius,
                                       discretizations,
                                       t_values)
    else: # planar polygon
        area = planar_polygon_area(vertices)
    
    return abs(area)

def poly_area_dispatch(vertices,
                       radius=None,
                       double threshold=1e-21,
                       int discretizations=500,
                       int n_rot=50,
                       int n_polygons=1):
    cdef:
        int polygon
        double[::1] t_values = np.linspace(0, 1, discretizations)

    if n_polygons == 1:
        area = poly_area(vertices=vertices,
                         t_values=t_values,
                         radius=radius,
                         threshold=threshold,
                         discretizations=discretizations,
                         n_rot=n_rot)
    else:
        area = np.empty(n_polygons)
        for polygon in range(n_polygons):
            area[polygon] = poly_area(vertices=vertices[polygon],
                                      t_values=t_values,
                                      radius=radius,
                                      threshold=threshold,
                                      discretizations=discretizations,
                                      n_rot=n_rot)
    return area
