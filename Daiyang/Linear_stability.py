import numpy as np
import matplotlib.pyplot as plt


# Hilbert Transform
def hilbert(u, k):
    sgn = np.sign(k)
    sgn[0] = 0
    u_hat = np.fft.fft(u)
    return np.fft.ifft(1j * sgn * u_hat).real


# Derivative of u
def spatial_derivatives(u, k):
    """Compute the derivatives int the Fourier space."""
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-k**2 * u_hat).real
    return np.array(ux), np.array(uxx)


def f(u, c, H, k):
    N = len(u)
    ux, uxx = spatial_derivatives(u, k)
    H_ux = hilbert(ux, k)
    H_uux = hilbert(u * ux, k)
    f = (-c * H_ux) - uxx + H_uux
    mid = N//2
    eq2 = u[mid] - u[0] - H
    return np.append(f, eq2)


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


def newton(H, *, N=2000, L=np.pi, tol=1e-12, n=30):
    dx = 2*L / N
    x = np.arange(-L, L, dx)
    k = np.fft.fftfreq(N, d=dx) * 2*np.pi

    # initial guess
    u = 0.5 * H * np.cos(x)
    c = 1.0

    # Newton iteration
    for _ in range(n):
        res = f(u, c, H, k)
        if np.linalg.norm(res, np.inf) < tol:
            break

        delta = np.linalg.solve(jacobian(u, c, k, H, f), -res)
        u += delta[:-1]
        c += delta[-1]

    return x, u, c, k


# L = -c H dx - dx² + H( U dx + Ux )
def L(v, U, Ux, c, k):
    vx = np.fft.ifft(1j * k * np.fft.fft(v)).real
    vxx = np.fft.ifft(-k**2 * np.fft.fft(v)).real
    H_vx = hilbert(vx, k)

    # non‑constant coefficients handled in physical space
    H_term = hilbert(U * vx + Ux * v, k)

    return -c * H_vx - vxx + H_term


def L_matrix(U, c, k):
    N = len(U)
    Ux, _ = spatial_derivatives(U, k)

    L_mat = np.zeros((N, N))

    for j in range(N):
        e_j = np.zeros(N)
        e_j[j] = 1.0  # standard basis vector
        L_mat[:, j] = L(e_j, U, Ux, c, k)
    return L_mat


x, U, c, k = newton(H=0.5, N=128)
Lmat1 = L_matrix(U, c, k)
eigvals = np.linalg.eigvals(Lmat1)

plt.figure(figsize=(6, 4))
plt.scatter(eigvals.real, eigvals.imag)
plt.xlabel(r'Re$(\lambda)$')
plt.ylabel(r'Im$(\lambda)$')
plt.show()
