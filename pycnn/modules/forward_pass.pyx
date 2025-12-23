import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline void relu_inplace(np.ndarray[DTYPE_t, ndim=2] arr):
    cdef Py_ssize_t i, j
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] < 0:
                arr[i, j] = 0

def forward_pass(np.ndarray[DTYPE_t, ndim=2] batch_x,
                 list activations,
                 list z_values,
                 dict weights,
                 dict biases,
                 list layers):

    activations[0] = batch_x
    current_input = batch_x

    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] z

    for i in range(len(layers)):
        z = current_input @ weights[f"w{i+1}"].T
        z += biases[f"b{i+1}"]
        z_values[i] = z
        relu_inplace(z)
        current_input = z
        activations[i + 1] = current_input

    z = current_input @ weights["wo"].T
    z += biases["bo"]
    z_values[len(layers)] = z

    cdef np.ndarray[DTYPE_t, ndim=2] exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    cdef np.ndarray[DTYPE_t, ndim=2] output = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return output
