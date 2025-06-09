import numpy as np
import matplotlib.pyplot as plt


# Hilbert Transform
def hilbert(u, k):
    sgn = np.sign(k)
    sgn[0] = 0
    u_hat = np.fft.fft(u)
    return np.fft.ifft(-1j * sgn * u_hat).real


# Derivative of u
def spatial_derivatives(u, k):
    """Compute the derivatives int the Fourier space."""
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-k**2 * u_hat).real
    return np.array(ux), np.array(uxx)


# Find f(u): [-cH(u_x) - u_xx + H(u*u_x), u(0) - u(-L) + H]
def f(u, c, H, k):
    N = len(u)
    ux, uxx = spatial_derivatives(u, k)
    H_ux = hilbert(ux, k)
    H_uux = hilbert(u * ux, k)
    f = (-c * H_ux) - uxx + H_uux
    mid = N//2
    eq2 = u[mid] - u[0] - H
    return np.append(f, eq2)


# Jacobian
def jacobian(u, c, k, H, func):
    delta = 1e-8
    N = len(u)
    f0 = func(u, c, H, k)
    J = np.empty((N + 1, N + 1))

    for i in range(N):
        du = np.zeros(N)
        du[i] = delta
        J[:, i] = (func(u + du, c, H, k) - f0) / delta

    J[:, -1] = (func(u, c + delta, H, k) - f0) / delta

    return J


def newton(H, *, N=128, L=20, tol=1e-12, n=30):
    dx = 2*L / N
    x = np.arange(-L, L, dx)
    k = np.fft.fftfreq(N, d=dx) * 2*np.pi

    # initial guess
    u = H/2 * np.cos(x)
    c = 1.0

    # Newton iteration
    for _ in range(n):
        res = f(u, c, H, k)
        if np.linalg.norm(res, np.inf) < tol:
            break

        delta = np.linalg.solve(jacobian(u, c, k, H, f), -res)
        u += delta[:-1]
        c += delta[-1]

    plt.plot(x, H/2 * np.cos(x), label='Initial guess')
    plt.plot(x, u, label='Newtons Solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.legend()
    plt.show()


H = 1
newton(H)
