"""Calculates the solution to B-O using Eulers method.

eulers.py

"""

import numpy as np
import matplotlib.pyplot as plt


# Need to test this function and work on it.
def hilbert_transform(u, k_sign):
    """Compute the hulbert transform by Fouriers method."""
    u_hat = np.fft.fft(u)
    H_hat = -1j * k_sign * u_hat
    return np.fft.ifft(H_hat).real


def spatial_derivatives(u, k):
    """Compute the derivatives int the Fourier space."""
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-k**2 * u_hat).real
    return ux, uxx


def f(u, dt, k, k_sign):
    """Return the B-O equation."""
    ux = spatial_derivatives(u, k)
    uxx = spatial_derivatives(ux, k)
    Hu_xx = hilbert_transform(uxx, k_sign)
    return u - dt * (Hu_xx + u * ux)


def eulers(u0):
    """Solve the Benjamin-Ono ODE given the initial condition of u0."""
    N = 512              # Number of spatial points
    L = 20               # Domain size [-L, L]
    T = 2.0              # Final time
    dt = 0.001           # Time step
    steps = int(T/dt)    # Number of time steps
    # Spatial grid
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers
    k_sign = np.sign(k)
    u = u0  # initial guess

    for n in range(steps):
        u = f(u, dt, k, k_sign)
    plt.plot(x, u, label='u(x, T)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title("Benjamin–Ono (Euler Method Approximation)")
    plt.grid(True)
    plt.legend()
    return u
