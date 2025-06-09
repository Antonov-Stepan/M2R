"""
Newton's Method for solving travelling wave solution

References:
"""

import numpy as np
import matplotlib.pyplot as plt


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


def f(u, L, c, amplitude):
    """Return the evaluation of the function"""
    N = len(u)
    X = np.linspace(-L, L, num=N, endpoint=False)
    dx = X[1] - X[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_sign = np.sign(k)
    ux, uxx = spatial_derivatives(u, k)
    H_ux = hilbert_transform(ux, k_sign)
    H_uux = hilbert_transform(u * ux, k_sign)
    f = (-c * H_ux) - uxx + H_uux
    # min is 0 and max is N/2 + 1
    max = (N//2) + 1
    # MAX - MIN
    constraint = u[max] - u[0] - amplitude
    return np.append(f, constraint)


def jacobian(u, L, c, amplitude, res):
    """Computes the Jacobian"""
    N = len(u)
    delta = 10**-7
    J = np.zeros((N+1, N+1))

    # w.r.t u
    for j in range(N):
        delta_uj = np.zeros(N)
        delta_uj += u
        delta_uj[j] += delta
        delta_res = f(delta_uj, L, c, amplitude)
        J[:, j] = (delta_res - res) / delta

    # w.r.t c
    delta_c_res = f(u, L, c + delta, amplitude)
    J[:, -1] = (delta_c_res - res) / delta

    return J


def newton_meth(ui, L, amp, c=1.0):
    """Solve the travelling wave solution with newton's method"""

    # initial guess
    u = np.copy(ui)
    res = f(u, L, c, amp)
    err = np.max(np.abs(res))

    # newton
    while err > 10**-10:
        J = jacobian(u, L, c, amp, res)
        corr = np.linalg.solve(-J, res)
        u += corr[:-1]
        c += corr[-1]
        res = f(u, L, c, amp)
        err = np.max(np.abs(res))

    return u, c


def plot(N, L, ui, u, c, amp):
    """
    Graphing initial guess and solution

    ui: initial guess
    u: solution
    """

    X = np.linspace(-L, L, num=N, endpoint=False)
    plt.plot(X, ui, label="Initial Guess")
    plt.plot(X, u, label="Solution")
    plt.title(f"Travelling wave solution with {N} grid points")
    plt.plot(X, f(u, L, c, amp)[:-1], label="f")
    plt.xlabel("X")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"c estimate:{c}")


def bifurcation(N, amp, n, u):
    """Create bifurcation diagram for amplitude and c"""

    ui = np.copy(u)
    c = 1.0
    step = amp/n
    c_values = np.zeros(n)
    h_values = np.zeros(n)
    L = np.pi
    X = np.linspace(-L, L, num=N, endpoint=False)
    plt.plot(X, ui, label="Initial Guess")
    plt.title("Travelling wave bifurcation")
    plt.xlabel("X")
    plt.ylabel("u")
    plt.grid(True)
    for i in range(n):
        amplitude = step * (i + 1)
        uii = np.copy(ui)
        ui, c = newton_meth(ui, L, amplitude, c=c)
        plot(N, L, uii, ui, c, amplitude)
        print(ui[N//2 + 1] - ui[0])
        plt.plot(X, ui, label=f"h = {amplitude}")
        # print(c)
        c_values[i] = c
        h_values[i] = amplitude
    plt.legend()
    plt.show()

    plt.plot(c_values, h_values)
    plt.scatter(c_values, h_values)
    plt.title("Bifurcation diagram")
    plt.xlabel("c")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.show()


def start():
    N = 512
    L = np.pi
    amp = 0.1
    X = np.linspace(-L, L, num=N, endpoint=False)
    ui = (amp/2)*np.cos(X)

    u, c = newton_meth(ui, L, amp)
    plot(N, L, ui, u, c, amp)
    bifurcation(N, amp, 10, ui)


start()
