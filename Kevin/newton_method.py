'''
Newton's Method for solving travelling wave solution

References:
'''

import numpy as np


def newton_meth():
    N = 10
    X = np.linspace(0, 2*np.pi, N)
    u = 0.01*np.cos(X)