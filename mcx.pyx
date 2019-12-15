"""
Author: Isaac Drachman
Date:   12/15/2019

Cython implementation of the discrete steps for generating stochastic processes.
"""

import numpy as np
cimport numpy as np

def heston_step(float S0, float nu0, float kappa, float theta, float xi, float rho, float dt, float r, float q, int num_paths, int num_steps,
                np.ndarray[np.float64_t, ndim=2] dW_S, np.ndarray[np.float64_t, ndim=2] dW_X):
    cdef np.ndarray[np.float64_t, ndim=2] paths = np.zeros((num_paths,num_steps)) + S0
    cdef np.ndarray[np.float64_t, ndim=2] dW_nu = rho*dW_S + np.sqrt(1 - rho**2)*dW_X
    cdef np.ndarray[np.float64_t, ndim=1] nu = np.zeros((num_paths)) + nu0
    cdef int t = 1
    for t in range(1,num_steps):
        # Since the variance process is discrete, it can take on negative values
        # Use full truncation, i.e. v_t <- max(v_t,0)
        nu = nu + kappa*(theta - np.maximum(nu,0))*dt + xi*np.sqrt(np.maximum(nu,0)*dt)*dW_nu[:,t]
        paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + np.sqrt(np.maximum(nu,0)*dt)*dW_S[:,t])
    return paths

def oujd_step(float S0, float mu, float theta, float sigma, float jump, float dt, int num_paths, int num_steps,
              np.ndarray[np.float64_t, ndim=2] noise, np.ndarray[np.float64_t, ndim=2] poiss):
    cdef np.ndarray[np.float64_t, ndim=2] paths = np.zeros((num_paths,num_steps)) + S0
    cdef int t = 1
    for t in range(1,num_steps):
        paths[:,t] = paths[:,t-1] + theta*(mu - paths[:,t-1])*dt + sigma*np.sqrt(dt)*noise[:,t] + jump*poiss[:,t]
    return paths

def gbmjd_step(float S0, float r, float q, float sigma, float jump, float dt, int num_paths, int num_steps,
               np.ndarray[np.float64_t, ndim=2] noise, np.ndarray[np.float64_t, ndim=2] poiss):
    cdef np.ndarray[np.float64_t, ndim=2] paths = np.zeros((num_paths,num_steps)) + S0
    cdef int t = 1
    for t in range(1,num_steps):
        paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + sigma*np.sqrt(dt)*noise[:,t] + jump*poiss[:,t])
    return paths
