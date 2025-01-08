# lorenz96_cython.pyx

import numpy as np
cimport numpy as cnp
from libc.math cimport fmod

def rk4_cython(f, double t, cnp.ndarray[double, ndim=1] x, cnp.ndarray[double, ndim=1] p, double dt):
    cdef:
        int J = x.shape[0]
        cnp.ndarray[double, ndim=1] k1 = np.zeros(J, dtype=np.float64)
        cnp.ndarray[double, ndim=1] k2 = np.zeros(J, dtype=np.float64)
        cnp.ndarray[double, ndim=1] k3 = np.zeros(J, dtype=np.float64)
        cnp.ndarray[double, ndim=1] k4 = np.zeros(J, dtype=np.float64)
        cnp.ndarray[double, ndim=1] xt = np.zeros(J, dtype=np.float64)
    
    # Compute Runge-Kutta steps
    k1[:] = f(t, x, p)
    k2[:] = f(t + dt / 2, x + k1 * dt / 2, p)
    k3[:] = f(t + dt / 2, x + k2 * dt / 2, p)
    k4[:] = f(t + dt, x + k3 * dt, p)
    
    # Combine to compute next state
    xt[:] = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return xt


def lorenz96_cython(double t, cnp.ndarray[double, ndim=1] x, cnp.ndarray[double, ndim=1] p):
    cdef:
        int J = x.shape[0]
        cnp.ndarray[double, ndim=1] dxdt = np.zeros_like(x)
        int i
    for i in range(J):
        dxdt[i] = (x[(i + 1) % J] - x[i - 2]) * x[i - 1] - x[i] + p[i]
    return dxdt

def lorenz96_jacobian_cython(double t, cnp.ndarray[double, ndim=1] x, cnp.ndarray[double, ndim=1] p):
    cdef:
        int J = x.shape[0]
        cnp.ndarray[double, ndim=2] jac = np.zeros((J, J))
    for i in range(J):
        jac[i, (i - 2) % J] = -x[i - 1]
        jac[i, (i - 1) % J] = x[(i + 1) % J] - x[i - 2]
        jac[i, i] = -1
        jac[i, (i + 1) % J] = x[i - 1]
    return jac
