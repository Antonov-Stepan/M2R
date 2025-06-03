"""
Newton's Method for solving travelling wave solution

References:
"""

import numpy as np


def hilbert_transform():
    """Calculate the hilbert transform"""
    pass


def f(u):
    """Return the evaluation of the function"""
    pass


def newton_meth():
    """Solve the travelling wave solution with newton's method"""
    N = 10
    X = np.linspace(0, 2*np.pi, num=N, endpoint=False)
    u = 0.01*np.cos(X)
    b = f(u)
    err = np.max(np.abs(b))
    while err > 10**-10:
        #Jacobian
        J = np.zeros((N, 2))
        del1 = 10**-7
        del2 = 10**-7
        delta = 10**-7
        J[:, 1] = (f(u + del1) - b)/delta
        J[:, 2] = (f(u + del2)- b)/delta
        corr = np.linalg.solve(-J, b)
        u += corr
        b = f(u)
        err = np.max(np.abs(b))
    