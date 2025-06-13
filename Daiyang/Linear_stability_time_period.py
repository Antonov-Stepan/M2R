import numpy as np
import matplotlib.pyplot as plt

# ——— Helpers ———


def hilbert(u, k):
    sgn = np.sign(k)
    sgn[0] = 0
    u_hat = np.fft.fft(u)
    return np.fft.ifft(1j * sgn * u_hat).real


def spatial_derivatives(u, k):
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-k**2 * u_hat).real
    return ux, uxx


def f(u, c, H, k):
    ux, uxx = spatial_derivatives(u, k)
    H_ux = hilbert(ux,  k)
    H_uux = hilbert(u*ux, k)
    mid = len(u)//2
    eq2 = u[mid] - u[0] - H
    return np.append((-c*H_ux) - uxx + H_uux, eq2)


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


def Lv(v, U, Ux, c, k):

    v_hat = np.fft.fft(v)
    vx = np.fft.ifft(1j*k * v_hat).real
    vxx = np.fft.ifft(-k**2 * v_hat).real

    return c*vx - hilbert(vxx, k) - (Ux*v + U*vx)


def L_matrix(U, c, k):
    N = len(U)
    Ux, _ = spatial_derivatives(U, k)
    L = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1
        L[:, j] = Lv(e, U, Ux, c, k)
    return L


def rhs(u, c, k):
    ux, uxx = spatial_derivatives(u, k)
    return -c*hilbert(ux, k) - uxx + hilbert(u*ux, k)


def rk4_step(u, dt, rhs, c, k):
    k1 = rhs(u, c, k)
    k2 = rhs(u+0.5*dt*k1, c, k)
    k3 = rhs(u+0.5*dt*k2, c, k)
    k4 = rhs(u + dt*k3, c, k)
    return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def measure_period_fft(a, dt, T_guess):

    a = a - a.mean()
    A = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(a.size, dt)

    # index of the bin that should contain 1/T_guess
    i0 = int(round((1.0/T_guess) / freqs[1]))

    # guard against the ends of the spectrum
    i_min = max(i0 - 1, 1)
    i_max = min(i0 + 1, len(A)-1)

    idx = i_min + np.argmax(A[i_min:i_max+1])
    return 1.0 / freqs[idx]


# travelling wave
x, U, c, k = newton(H=0.5, N=128)

# eigenvalues and eigenvectors
Lmat = L_matrix(U, c, k)
eigvals, eigvecs = np.linalg.eig(Lmat)


modes = np.where(np.abs(eigvals.imag) > 1e-8)[0]
modes = modes[np.argsort(np.abs(eigvals.imag[modes]))]


dt = 1e-4
eps = 1e-2
omega_cut = 500
max_steps = 80_000

ω_list, T_thy_list, T_meas_list = [], [], []

for idx in modes:
    ω = np.imag(eigvals[idx])
    if abs(ω) >= omega_cut:
        continue

    V = eigvecs[:, idx]
    T_thy = 2*np.pi / abs(ω)
    n_steps = int(10*T_thy / dt)
    if n_steps > max_steps:
        continue

# initial guess
    u = U + eps*(V + np.conj(V))
    a = np.empty(n_steps+1)
    nom = np.linalg.norm(np.imag(V))**2

    for n in range(n_steps+1):
        a[n] = np.dot(u-U, np.imag(V)) / nom
        if n < n_steps:
            u = rk4_step(u, dt, rhs, c, k)

    T_meas = measure_period_fft(a, dt, T_thy)

    ω_list.append(abs(ω))
    T_thy_list.append(T_thy)
    T_meas_list.append(T_meas)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(ω_list, T_thy_list, 'o-', label='Theory $2π/|ω|$')
plt.plot(ω_list, T_meas_list, 'x-', label='Measured (FFT)')
plt.xlabel(r'$|ω|$')
plt.ylabel('Period $T$')
plt.title('Predicted vs Measured Periods')
plt.legend()
plt.tight_layout()
plt.show()
