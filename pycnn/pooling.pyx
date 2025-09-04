import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative index wraparound
def max_pooling(np.ndarray[double, ndim=2] feature_map):
    cdef int stride = 2
    cdef int size = 2
    cdef int h = feature_map.shape[0]
    cdef int w = feature_map.shape[1]
    cdef int out_h = (h - size) // stride + 1
    cdef int out_w = (w - size) // stride + 1
    cdef np.ndarray[double, ndim=2] pooled = np.zeros((out_h, out_w), dtype=np.float64)
    
    cdef int i, j, ii, jj
    cdef double max_val
    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            max_val = feature_map[i, j]
            for ii in range(i, i + size):
                for jj in range(j, j + size):
                    if feature_map[ii, jj] > max_val:
                        max_val = feature_map[ii, jj]
            pooled[i // stride, j // stride] = max_val
    return pooled
