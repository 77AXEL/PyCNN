import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def max_pooling(np.ndarray[DTYPE_t, ndim=2] feature_map):
    cdef int h = feature_map.shape[0]
    cdef int w = feature_map.shape[1]
    cdef int new_h = h // 2
    cdef int new_w = w // 2
    cdef np.ndarray[DTYPE_t, ndim=2] pooled = np.zeros((new_h, new_w), dtype=DTYPE)

    cdef int i, j
    cdef int ii, jj
    cdef DTYPE_t val

    for i in range(new_h):
        for j in range(new_w):
            val = feature_map[i*2, j*2]
            for ii in range(2):
                for jj in range(2):
                    if feature_map[i*2 + ii, j*2 + jj] > val:
                        val = feature_map[i*2 + ii, j*2 + jj]
            pooled[i, j] = val

    return pooled
