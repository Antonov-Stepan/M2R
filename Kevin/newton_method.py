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


def f(u, k, k_sign, c, amplitude):
    """Return the evaluation of the function"""
    N = len(u)
    ux, uxx = spatial_derivatives(u, k)
    H_ux = hilbert_transform(ux, k_sign)
    H_uux = hilbert_transform(u * ux, k_sign)
    f = (-c * H_ux) - uxx + H_uux
    # min is 0 and max is N/2 + 1
    max = (N//2) + 1
    # MAX - MIN
    constraint = u[max] - u[0] - amplitude
    return np.append(f, constraint)


def jacobian(u, k, k_sign, c, amplitude, res):
    """Computes the Jacobian"""
    N = len(u)
    delta = 10**-7
    J = np.zeros((N+1, N+1))

    # w.r.t u
    for j in range(N):
        delta_uj = np.zeros(N)
        delta_uj += u
        delta_uj[j] += delta
        delta_res = f(delta_uj, k, k_sign, c, amplitude)
        J[:, j] = (delta_res - res) / delta

    # w.r.t c
    delta_c_res = f(u, k, k_sign, c + delta, amplitude)
    J[:, -1] = (delta_c_res - res) / delta

    return J


def f2(u, c, amplitude):
    """Return the evaluation of the function"""
    # only have half the u, so replicate to make full, then return half
    # full u
    uf = np.concatenate((u, u[-2:0:-1]))
    L = np.pi
    N = (len(u) - 1) * 2
    X = np.linspace(-L, L, num=N, endpoint=False)
    dx = X[1] - X[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_sign = np.sign(k)
    ux, uxx = spatial_derivatives(uf, k)
    H_ux = hilbert_transform(ux, k_sign)
    H_uux = hilbert_transform(uf * ux, k_sign)
    f = (-c * H_ux) - uxx + H_uux
    f = f[0:(N//2 + 1)]
    constraint = u[-1] - u[0] - amplitude
    return np.append(f, constraint)


def jacobian2(u, c, amplitude, res):
    """Computes the Jacobian for half solution"""
    N = len(u)
    delta = 10**-7
    J = np.zeros((N+1, N+1))

    # w.r.t u
    for j in range(N):
        delta_uj = np.zeros(N)
        delta_uj += u
        delta_uj[j] += delta
        delta_res = f2(delta_uj, c, amplitude)
        J[:, j] = (delta_res - res) / delta

    # w.r.t c
    delta_c_res = f2(u, c + delta, amplitude)
    J[:, -1] = (delta_c_res - res) / delta

    return J


def newton_meth(N, L, amp):
    """Solve the travelling wave solution with newton's method"""

    # create grid
    X = np.linspace(-L, L, num=N, endpoint=False)
    dx = X[1] - X[0]

    # initial guess
    u = (amp/2)*np.cos(X)
    plt.plot(X, u, label="Initial Guess")
    c = 1.0
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_sign = np.sign(k)
    res = f(u, k, k_sign, c, amp)
    err = np.max(np.abs(res))

    # newton
    while err > 10**-10:
        J = jacobian(u, k, k_sign, c, amp, res)
        corr = np.linalg.solve(-J, res)
        u += corr[:-1]
        c += corr[-1]
        res = f(u, k, k_sign, c, amp)
        err = np.max(np.abs(res))

    plt.plot(X, u, label="Solution")
    plt.title(f"Travelling wave solution with {N} grid points")
    plt.xlabel("X")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    # plt.plot(X, f(u, k, k_sign, c, amp)[:-1], label="f")
    plt.show()
    print(f"c estimate:{c}")


def newton_meth2(N, u, amp, c=1.0):
    """Only solving half the solution but return full reflected solution"""

    # create grid
    # h = half
    Nh = N//2 + 1

    # initial guess
    ui = np.copy(u)
    ui = ui[0:Nh]
    res = f2(ui, c, amp)
    err = np.max(np.abs(res))

    # newton
    while err > 10**-10:
        J = jacobian2(ui, c, amp, res)
        corr = np.linalg.solve(-J, res)
        ui += corr[:-1]
        c += corr[-1]
        res = f2(ui, c, amp)
        err = np.max(np.abs(res))

    # final u
    uf = np.concatenate((ui, ui[-2:0:-1]))
    return uf, c


def plot(N, L, ui, u, c):
    """
    Graphing initial guess and solution

    ui: initial guess
    u: solution
    """
    X = np.linspace(-L, L, num=N, endpoint=False)
    plt.plot(X, ui, label="Initial Guess")
    plt.plot(X, u, label="Solution")
    plt.title(f"Travelling wave solution with {N} grid points, method 2")
    plt.xlabel("X")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    # plt.plot(X, f(u, k, k_sign, c, amp)[:-1], label="f")
    plt.show()
    print(f"c estimate:{c}")


def bifurcation(N, amp, n, u):
    """Create bifurcation diagram for amplitude and c"""

    ui = np.copy(u)
    c = 1.0
    step = amp/n
    c_values = np.zeros(n)
    h_values = np.zeros(n)
    for i in range(n):
        amplitude = step * (i + 1)
        ui, c = newton_meth2(N, ui, amplitude, c=c)
        # print(c)
        c_values[i] = c
        h_values[i] = amplitude

    plt.plot(c_values, h_values)
    plt.scatter(c_values, h_values)
    plt.title("Bifurcation diagram")
    plt.xlabel("c")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.show()


N = 512
L = np.pi
amp = 0.1
X = np.linspace(-L, L, num=N, endpoint=False)
ui = (amp/2)*np.cos(X)

newton_meth(N, L, amp)
u, c = newton_meth2(N, ui, amp)
print(len(ui))
plot(N, L, ui, u, c)
bifurcation(N, amp, 100, ui)
