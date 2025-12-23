import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void relu_derivative(DTYPE_t[:, :] dz, DTYPE_t[:, :] z_vals) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t rows = z_vals.shape[0]
    cdef Py_ssize_t cols = z_vals.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if z_vals[i, j] <= 0:
                dz[i, j] = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t[:] sum_axis0(DTYPE_t[:, :] arr):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(cols, dtype=DTYPE)
    cdef DTYPE_t[:] result_view = result
    
    for i in range(rows):
        for j in range(cols):
            result_view[j] += arr[i, j]
    
    return result_view

@cython.boundscheck(True) 
@cython.wraparound(True)
def backward_pass(output,
                  batch_y,
                  list activations,
                  list z_values,
                  dict gradients_w,
                  dict gradients_b,
                  dict weights,
                  list layers):
    """
    Backward pass with proper exception handling
    """
    cdef np.ndarray[DTYPE_t, ndim=2] output_arr
    cdef np.ndarray[DTYPE_t, ndim=2] batch_y_arr
    cdef np.ndarray[DTYPE_t, ndim=2] dL_dz_out
    cdef np.ndarray[DTYPE_t, ndim=2] dL_da
    cdef np.ndarray[DTYPE_t, ndim=2] dL_dz
    cdef np.ndarray[DTYPE_t, ndim=2] activation_arr
    cdef np.ndarray[DTYPE_t, ndim=2] z_val_arr
    cdef np.ndarray[DTYPE_t, ndim=2] weight_arr
    cdef int i
    cdef str weight_key, bias_key
    
    output_arr = np.asarray(output, dtype=np.float64)
    batch_y_arr = np.asarray(batch_y, dtype=np.float64)
    
    if output_arr.ndim != 2 or batch_y_arr.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got output: {output_arr.ndim}D, batch_y: {batch_y_arr.ndim}D")
    
    dL_dz_out = output_arr - batch_y_arr
    
    activation_arr = np.asarray(activations[-1], dtype=np.float64)
    if activation_arr.ndim != 2:
        raise ValueError(f"Expected 2D activation array, got {activation_arr.ndim}D")
    
    gradients_w["wo"][:] = dL_dz_out.T @ activation_arr
    gradients_b["bo"][:] = np.sum(dL_dz_out, axis=0)
    
    weight_arr = np.asarray(weights["wo"], dtype=np.float64)
    dL_da = dL_dz_out @ weight_arr
    
    for i in range(len(layers) - 1, -1, -1):
        weight_key = f"w{i+1}"
        bias_key = f"b{i+1}"
        
        z_val_arr = np.asarray(z_values[i], dtype=np.float64)
        if z_val_arr.ndim != 2:
            raise ValueError(f"Expected 2D z_values array at layer {i}, got {z_val_arr.ndim}D")
        
        dL_dz = dL_da.copy()
        dL_dz[z_val_arr <= 0] = 0.0
        
        activation_arr = np.asarray(activations[i], dtype=np.float64)
        if activation_arr.ndim != 2:
            raise ValueError(f"Expected 2D activation array at layer {i}, got {activation_arr.ndim}D")
        
        gradients_w[weight_key][:] = dL_dz.T @ activation_arr
        gradients_b[bias_key][:] = np.sum(dL_dz, axis=0)
        
        if i > 0:
            weight_arr = np.asarray(weights[weight_key], dtype=np.float64)
            dL_da = dL_dz @ weight_arr