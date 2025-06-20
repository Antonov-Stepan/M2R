
import numpy as np
import matplotlib.pyplot as plt


# Hilbert Transform
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


# Find f(u): [-cH(u_x) - u_xx + H(u*u_x), u(0) - u(-L) + H]
def f(u, c, H, k):
    N = len(u)
    ux, uxx = spatial_derivatives(u, k)
    k_sign = np.sign(k)
    k_sign[0] = 0
    H_ux = hilbert_transform(ux, k_sign)
    H_uux = hilbert_transform(u * ux, k_sign)
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


def newton(H, *, N=128, L=np.pi, tol=1e-12, n=30):
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

    return x, u, c, k


# L = c dx - H(dx^2) - (U dx + Ux) for u = U + \epsilon e^(\lambda t) v
def Lv(v, U, Ux, c, k):

    v_hat = np.fft.fft(v)
    vx = np.fft.ifft(1j*k * v_hat).real
    vxx = np.fft.ifft(-k**2 * v_hat).real

    k_sign = np.sign(k)
    k_sign[0] = 0

    H_vxx = hilbert_transform(vxx, k_sign)
    Uv_x = Ux * v + U * vx

    return c * vx - H_vxx - Uv_x


def L_matrix(U, c, k):
    N = len(U)
    Ux, _ = spatial_derivatives(U, k)
    L = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1
        L[:, j] = Lv(e, U, Ux, c, k)
    return L


x, U, c, k = newton(H=0.5, N=128)
Lmat = L_matrix(U, c, k)
eigvals, eigvecs = np.linalg.eig(Lmat)

# Plot of eigenvectors
indices = np.where((np.abs(eigvals.imag) < 100) & (eigvals.imag != 0))[0]

plt.figure(figsize=(10, 6))
for i in indices:
    v = eigvecs[:, i]
    plt.plot(x, v.real, label=f"Im(λ)={eigvals[i].imag}")

plt.xlabel("x")
plt.ylabel("Eigenvector component")
plt.title("Eigenvectors with |Im(λ)| < 100")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot of eigenvectors with imag part of eigenvalue = 0
indices = np.where((eigvals.imag == 0))[0]

plt.figure(figsize=(10, 6))
for i in indices:
    v = eigvecs[:, i]
    plt.plot(x, v.real)

plt.xlabel("x")
plt.ylabel("Eigenvector component")
plt.title("Eigenvectors with Im(λ) = 0 ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot of eigenvalues
plt.figure(figsize=(6, 4))
plt.scatter(eigvals.real, eigvals.imag, s=14)
plt.axvline(0, color='k', lw=0.8)
plt.xlabel(r'Re$(\lambda)$')
plt.ylabel(r'Im$(\lambda)$')
plt.show()
