"""
Newton's Method for solving travelling wave solution

References:
Parity of Hilbert Transform:
https://www.diva-portal.org/smash/get/diva2:872439/FULLTEXT02 pg14
Hilbert transform of hilbert transform:
https://en.wikipedia.org/wiki/Hilbert_transform
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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


def f2(u, c, amplitude, L):
    """
    Return the evaluation of the function
    u : function on half the domain
    returns values evaluated on half the domain, with constraint appended
    """
    # only have half the u, so replicate to make full: uf
    # then return half of the full u
    uf = np.concatenate((u, u[-2:0:-1]))
    uf_hat = np.fft.fft(uf)
    uf_hat[0] = 0
    uf = np.fft.ifft(uf_hat).real

    # L = np.pi
    N = (len(u) - 1) * 2
    X = np.linspace(-L, L, num=N, endpoint=False)
    dx = X[1] - X[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_sign = np.sign(k)

    ux, uxx = spatial_derivatives(uf, k)
    H_ux = hilbert_transform(ux, k_sign)
    H_uux = hilbert_transform(uf * ux, k_sign)
    fu = (-c * H_ux) - uxx + H_uux
    fu = fu[0:(N//2 + 1)]
    constraint = u[-1] - u[0] - amplitude

    return np.append(fu, constraint)


def jacobian2(u, c, amplitude, res, L):
    """Computes the Jacobian for half solution"""
    N = len(u)
    delta = 10**-7
    J = np.zeros((N+1, N+1))

    # w.r.t u
    for j in range(N):
        delta_uj = np.zeros(N)
        delta_uj += u
        delta_uj[j] += delta
        delta_res = f2(delta_uj, c, amplitude, L)
        J[:, j] = (delta_res - res) / delta

    # w.r.t c
    delta_c_res = f2(u, c + delta, amplitude, L)
    J[:, -1] = (delta_c_res - res) / delta

    return J


def newton_meth2(u, amplitude, c, L):
    """Solving half the solution but return full reflected solution"""

    N = len(u)
    N_half = N//2 + 1

    # initial guess
    ui = np.copy(u)
    ui = ui[0:N_half]
    res = f2(ui, c, amplitude, L)
    err = np.max(np.abs(res))

    # newton
    while err > 10**-10:
        J = jacobian2(ui, c, amplitude, res, L)
        correction = np.linalg.solve(-J, res)
        ui += correction[:-1]
        c += correction[-1]
        res = f2(ui, c, amplitude, L)
        err = np.max(np.abs(res))

    # full u
    uf = np.concatenate((ui, ui[-2:0:-1]))

    return uf, c


def plot(N, L, ui, u, c, amp):
    """
    Graphing initial guess and solution

    ui: initial guess
    u: solution
    """
    X = np.linspace(-L, L, num=N, endpoint=False)

    # plot initial guess and solution
    plt.figure(figsize=(10, 5))
    plt.plot(X, ui, label="Initial Guess")
    plt.plot(X, u, label="Solution: u")

    # f2 takes half of domain, and returns half of solutions
    u_half = u[0:N//2+1]
    fu_half = f2(u_half, c, amp, L)[:-1]
    fu_full = np.concatenate((fu_half, fu_half[-2:0:-1]))
    plt.plot(X, fu_full, label="f(u)")
    plt.title(f"Travelling wave solution with {N} grid points")
    plt.xlabel("X")
    plt.ylabel("u")

    # plot analytical solution along with solution to commpare
    t = 0
    analytical = (amp)/(1 + c**2 * (X-c*t)**2)
    # plt.plot(X, analytical, label="Analytical solution")

    plt.legend()
    plt.grid(True)
    plt.show()
    # print(f"c estimate:{c}")


def bifurcation(N, amp, n, c, L):
    """Create bifurcation diagram for amplitude and c"""
    step = amp/n
    c_values = np.zeros(n)
    h_values = np.zeros(n)
    solutions = []
    X = np.linspace(-L, L, num=N, endpoint=False)

    # initial guess
    ui = ((step/2)*np.cos((X)*(np.pi/L)))
    # t = 0
    # ui = (amp)/(1 + c**2 * (X-c*t)**2)
    # plt.plot(X, ui, label="Initial Guess")
    plt.figure(figsize=(5, 5))
    plt.title("Travelling wave solutions")
    plt.xlabel("X")
    plt.ylabel("u")
    plt.grid(True)

    for i in range(n):
        amplitude = step * (i + 1)
        ui, c = newton_meth2(ui, amplitude, c, L)
        solutions += [ui]

        plt.plot(X, ui, label=f"h = {amplitude}")
        t = 0
        analytical = (amplitude)/(1 + c**2 * (X-c*t)**2)
        # plt.plot(X, analytical, label=f"Analytical solution, h={amplitude}")
        # u_half = ui[0:N//2+1]
        # fu_half = f2(u_half, c, amplitude)[:-1]
        # fu_full = np.concatenate((fu_half, fu_half[-2:0:-1]))
        # plt.plot(X, fu_full, label=f"f(u) for amp = {amplitude}")
        c_values[i] = c
        h_values[i] = amplitude
    # plt.legend(fontsize="x-small")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(c_values, h_values)
    plt.scatter(c_values, h_values)
    plt.title("amplitude and wave speed bifurcation diagram")
    plt.xlabel("c")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.show()
    # solutions_arr = np.array(solutions)
    # store(solutions_arr)


def store(u):
    with open(r"Kevin\solutions.pkl", "wb") as file:
        pickle.dump(u, file)


def start():
    N = 256
    L = 50
    amp = 0.5
    c = -np.pi/L
    X = np.linspace(-L, L, num=N, endpoint=False)
    ui = ((amp/2)*np.cos(X*(np.pi/L)))
    # t = 0
    # ui = (amp)/(1 + c**2 * (X-c*t)**2)
    # with open("Kevin\solutions.pkl", "rb") as file:
    #     a = pickle.load(file)
    # print(a)
    # print(a[2])
    # newton_meth(N, L, amp)
    u, c = newton_meth2(ui, amp, c, L)
    print(c)
    plot(N, L, ui, u, c, amp)
    bifurcation(N, amp, 50, c, L)


start()
