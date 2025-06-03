"""Calculates the solution to B-O using Eulers method.

eulers.py

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def hilbert_transform(u, k_sign):
    """Compute the Hilbert transform by Fouriers method."""
    u_hat = np.fft.fft(u)
    H_hat = -1j * k_sign * u_hat
    return np.fft.ifft(H_hat).real


def spatial_derivatives(u, k):
    """Compute the derivatives int the Fourier space."""
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-k**2 * u_hat).real
    return np.array(ux), np.array(uxx)


def f(u, k, k_sign):
    """Return the B-O equation."""
    ux, uxx = spatial_derivatives(u, k)
    Hu_xx = hilbert_transform(uxx, k_sign)
    return -(Hu_xx + u * ux)


def eulers(u0):
    """Solve the Benjamin-Ono ODE given the initial condition of u0."""
    N = 100              # Number of spatial points
    L = 1.5             # Domain size [-L, L]. Ensure the function is periodic on this domain.
    T = 10            # Final time
    dt = 0.0001           # Time step
    steps = int(T/dt)    # Number of time steps
    # Spatial grid
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers (double check convetion)
    k_sign = np.sign(k)
    u = [u0(x, L)]  # condition
    fig, ax = plt.subplots()
    line, = ax.plot(x, u[0])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    title = ax.set_title("t = 0.00")

    def update(frame):
        u[0] = u[0] + dt * f(u[0], k, k_sign)
        line.set_ydata(u[0])
        title.set_text(f"t = {frame * dt:.2f}")
        return line, title
    frame_interval = 1000
    ani = animation.FuncAnimation(fig, update, frames=range(0, steps, frame_interval), interval=30, repeat=False)
    plt.show()



def u0(x, L):
    return 0.5 * np.cos(np.pi * x / L)


eulers(u0)
