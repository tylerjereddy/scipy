# cython: cpow=True

"""
Directed Hausdorff Code

.. versionadded:: 0.19.0

"""
#
# Copyright (C)  Tyler Reddy, Richard Gowers, and Max Linke, 2016
#
# Distributed under the same BSD license as Scipy.
#

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, INFINITY

np.import_array()

__all__ = ['directed_hausdorff']

@cython.boundscheck(False)
def directed_hausdorff(const double[:,::1] ar1, const double[:,::1] ar2, seed=0):

    cdef double cmax, cmin, d = 0
    cdef Py_ssize_t N1 = ar1.shape[0]
    cdef Py_ssize_t N2 = ar2.shape[0]
    cdef int data_dims = ar1.shape[1]
    cdef Py_ssize_t i, j, k, w, z
    cdef Py_ssize_t j_store = 0, i_ret = 0, j_ret = 0
    cdef int max_pre
    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] resort1, resort2
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] inf_array
    cdef np.float64_t[:] tmds

    # shuffling the points in each array generally increases the likelihood of
    # an advantageous break in the inner search loop and never decreases the
    # performance of the algorithm
    if isinstance(seed, np.random.Generator):
        # Generator passthrough
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    resort1 = np.arange(N1, dtype=np.int64)
    resort2 = np.arange(N2, dtype=np.int64)
    rng.shuffle(resort1)
    rng.shuffle(resort2)
    ar1 = np.asarray(ar1)[resort1]
    ar2 = np.asarray(ar2)[resort2]
    inf_array = np.empty(N1, dtype=np.float64)
    tmds = inf_array

    for i in range(N1):
        tmds[i] = INFINITY

    cmax = 0
    if N1 + N2 > 2000:
        max_pre = 100
        if max_pre > N2:
            max_pre = N2

        for w in range(max_pre):
            cmin = INFINITY
            for z in range(N1):
                d = 0
                for k in range(data_dims):
                    d += (ar2[w, k] - ar1[z, k])**2
                if d < tmds[z]:
                    tmds[z] = d
                if d < cmin:
                    cmin = d
                if d < cmax:
                    break
            if cmin >= cmax:
                cmax = cmin

    cmax = 0
    for i in range(N1):
        if tmds[i] < cmax:
            continue
        cmin = np.inf
        for j in range(N2):
            d = 0
            # faster performance with square of distance
            # avoid sqrt until very end
            for k in range(data_dims):
                d += (ar1[i, k] - ar2[j, k])**2
            if d < cmax: # break out of `for j` loop
                break

            if d < cmin: # always true on first iteration of for-j loop
                cmin = d
                j_store = j

        # Note: The reference paper by A. A. Taha and A. Hanbury has this line
        # (Algorithm 2, line 16) as:
        #
        # if cmin > cmax:
        #
        # That logic is incorrect, as cmin could still be np.inf if breaking early.
        # The logic here accounts for that case.
        if cmin >= cmax and d >= cmax:
            cmax = cmin
            i_ret = i
            j_ret = j_store

    return (sqrt(cmax), resort1[i_ret], resort2[j_ret])
