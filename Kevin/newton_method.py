"""
Newton's Method for solving travelling wave solution

References:
"""

import numpy as np


def hilbert_transform():
    """Calculate the hilbert transform"""
    pass


def f(u, c=1.0):
    """Return the evaluation of the function"""
    # need this to be derivative of u
    u_x = 0
    H_u_x = hilbert_transform(u_x)
    return -c * u + 0.5 * u**2 + H_u_x


def jacobian(N, u):
    """Computes the Jacobian"""
    delta = 10**-7
    J = np.zeros((N, N))
    for j in range(N):
        delta_u_j = np.zeros(N)
        delta_u_j[j] = delta
        J[:, j] = (f(u + delta_u_j) - f(u)) / delta
    return J


def newton_meth():
    """Solve the travelling wave solution with newton's method"""
    N = 10
    # create grid
    X = np.linspace(0, 2*np.pi, num=N, endpoint=False)
    # initial guess
    u = 0.01*np.cos(X)
    b = f(u)
    err = np.max(np.abs(b))
    while err > 10**-10:
        J = jacobian(N, u)
        corr = np.linalg.solve(-J, b)
        u += corr
        b = f(u)
        err = np.max(np.abs(b))
