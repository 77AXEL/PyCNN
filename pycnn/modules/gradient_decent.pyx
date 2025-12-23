import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def gradient_decent(
    dict weights,
    dict biases,
    dict gradients_w,
    dict gradients_b,
    double learning_rate,
    str optimizer,
    dict m,
    dict v,
    double beta1,
    double beta2,
    double eps,
    int t):

    cdef str key
    cdef np.ndarray grad_w, weight, m_w, v_w
    cdef np.ndarray grad_b, bias, m_b, v_b
    cdef np.ndarray m_hat, v_hat, update
    cdef double bias_correction_1, bias_correction_2
    cdef double grad_norm, max_grad_norm = 5.0
    
    if optimizer == "sgd":
        for key in gradients_w:
            grad_w = gradients_w[key]
            weight = weights[key]
            
            grad_norm = np.linalg.norm(grad_w)
            if grad_norm > max_grad_norm:
                grad_w = grad_w * (max_grad_norm / grad_norm)
            
            if np.any(np.isnan(grad_w)) or np.any(np.isinf(grad_w)):
                print(f"Warning: NaN/Inf gradient detected in {key}, skipping update")
                continue
            
            weights[key] = weight - learning_rate * grad_w
        
        for key in gradients_b:
            grad_b = gradients_b[key]
            bias = biases[key]
            
            grad_norm = np.linalg.norm(grad_b)
            if grad_norm > max_grad_norm:
                grad_b = grad_b * (max_grad_norm / grad_norm)
            
            if np.any(np.isnan(grad_b)) or np.any(np.isinf(grad_b)):
                print(f"Warning: NaN/Inf gradient detected in bias {key}, skipping update")
                continue
                
            biases[key] = bias - learning_rate * grad_b
    
    elif optimizer == "adam":
        if t <= 0:
            t = 1
            
        bias_correction_1 = 1.0 - beta1 ** t
        bias_correction_2 = 1.0 - beta2 ** t
        
        if abs(bias_correction_1) < 1e-8:
            bias_correction_1 = 1e-8
        if abs(bias_correction_2) < 1e-8:
            bias_correction_2 = 1e-8
        
        for key in gradients_w:
            grad_w = gradients_w[key]
            weight = weights[key]
            
            grad_norm = np.linalg.norm(grad_w)
            if grad_norm > max_grad_norm:
                grad_w = grad_w * (max_grad_norm / grad_norm)
            
            if np.any(np.isnan(grad_w)) or np.any(np.isinf(grad_w)):
                print(f"Warning: NaN/Inf gradient detected in {key}, skipping update")
                continue
            
            if key not in m:
                m[key] = np.zeros_like(grad_w)
            if key not in v:
                v[key] = np.zeros_like(grad_w)
                
            m_w = m[key]
            v_w = v[key]
            
            m_w *= beta1
            m_w += (1.0 - beta1) * grad_w
            
            v_w *= beta2
            v_w += (1.0 - beta2) * (grad_w * grad_w)
            
            m_hat = m_w / bias_correction_1
            v_hat = v_w / bias_correction_2
            
            v_hat = np.maximum(v_hat, eps * eps)
            
            denominator = np.sqrt(v_hat) + eps
            update = learning_rate * m_hat / denominator
            
            if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                print(f"Warning: NaN/Inf update detected in {key}, skipping update")
                continue
                
            weights[key] = weight - update
        
        for key in gradients_b:
            grad_b = gradients_b[key]
            bias = biases[key]
            
            grad_norm = np.linalg.norm(grad_b)
            if grad_norm > max_grad_norm:
                grad_b = grad_b * (max_grad_norm / grad_norm)
            
            if np.any(np.isnan(grad_b)) or np.any(np.isinf(grad_b)):
                print(f"Warning: NaN/Inf gradient detected in bias {key}, skipping update")
                continue
            
            if key not in m:
                m[key] = np.zeros_like(grad_b)
            if key not in v:
                v[key] = np.zeros_like(grad_b)
                
            m_b = m[key]
            v_b = v[key]
            
            m_b *= beta1
            m_b += (1.0 - beta1) * grad_b
            
            v_b *= beta2
            v_b += (1.0 - beta2) * (grad_b * grad_b)
            
            m_hat = m_b / bias_correction_1
            v_hat = v_b / bias_correction_2
            
            v_hat = np.maximum(v_hat, eps * eps)
            
            denominator = np.sqrt(v_hat) + eps
            update = learning_rate * m_hat / denominator
            
            if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                print(f"Warning: NaN/Inf update detected in bias {key}, skipping update")
                continue
                
            biases[key] = bias - update
    
    return weights, biases, m, v, t