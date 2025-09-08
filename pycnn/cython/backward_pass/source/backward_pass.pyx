# backward_pass.pyx
# distutils: language = c++

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# ReLU derivative: inplace mask
cdef inline void relu_derivative(np.ndarray[DTYPE_t, ndim=2] dz,
                                 np.ndarray[DTYPE_t, ndim=2] z_vals):
    cdef Py_ssize_t i, j
    for i in range(z_vals.shape[0]):
        for j in range(z_vals.shape[1]):
            if z_vals[i, j] <= 0:
                dz[i, j] = 0.0

# Sum across axis=0 (like np.sum(arr, axis=0))
cdef np.ndarray[DTYPE_t, ndim=1] sum_axis0(np.ndarray[DTYPE_t, ndim=2] arr):
    cdef Py_ssize_t i, j
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(arr.shape[1], dtype=DTYPE)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[j] += arr[i, j]
    return result

# Backward pass
def backward_pass(np.ndarray output,
                  np.ndarray batch_y,
                  list activations,
                  list z_values,
                  dict gradients_w,
                  dict gradients_b,
                  dict weights,
                  list layers):
    """
    Cython backward pass function with automatic dtype casting.
    Accepts int/long arrays as input and casts them to float64 internally.
    """

    cdef np.ndarray[DTYPE_t, ndim=2] output_f = np.asarray(output, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] batch_y_f = np.asarray(batch_y, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] dL_dz_out
    cdef np.ndarray[DTYPE_t, ndim=2] dL_da
    cdef np.ndarray[DTYPE_t, ndim=2] dL_dz
    cdef int i

    # Output error
    dL_dz_out = output_f - batch_y_f

    # Gradients for output layer
    gradients_w["wo"][:] = dL_dz_out.T @ np.asarray(activations[-1], dtype=np.float64)
    gradients_b["bo"][:] = sum_axis0(dL_dz_out)

    dL_da = dL_dz_out @ np.asarray(weights["wo"], dtype=np.float64)

    # Hidden layers (backward)
    for i in range(len(layers) - 1, -1, -1):
        weight_key = f"w{i+1}"
        bias_key = f"b{i+1}"

        # Compute dL_dz = dL_da * ReLU'(z)
        dL_dz = dL_da.copy()
        relu_derivative(dL_dz, np.asarray(z_values[i], dtype=np.float64))

        # Gradients
        gradients_w[weight_key][:] = dL_dz.T @ np.asarray(activations[i], dtype=np.float64)
        gradients_b[bias_key][:] = sum_axis0(dL_dz)

        if i > 0:
            dL_da = dL_dz @ np.asarray(weights[weight_key], dtype=np.float64)

    return gradients_w, gradients_b
