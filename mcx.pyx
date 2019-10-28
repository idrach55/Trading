import numpy as np
cimport numpy as np

def GBM(float S0, float r, float q, float sigma, int num_paths, int num_steps, float dt):
    cdef np.ndarray[np.float64_t, ndim=2] rands = np.random.normal(0,sigma,size=(num_paths, num_steps))
    return S0*(1 + rands).cumprod(axis=1)
