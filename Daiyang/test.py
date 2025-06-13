import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq


def hilbert(u, k):
    sgn = np.sign(k)
    sgn[0] = 0
    return ifft(-1j * sgn * fft(u)).real


def spatial_derivatives(u, k):
    u_hat = fft(u)
    ux = ifft(1j * k * u_hat).real
    uxx = ifft(-k**2 * u_hat).real
    return ux, uxx

# Newton solve for travelling wave


def f_res(u, c, H, k):
    ux, uxx = spatial_derivatives(u, k)
    return np.concatenate([
        -c*ux + hilbert(uxx, k) + u*ux,
        [u[len(u)//2] - u[0] - H]
    ])


def jacobian(u, c, H, k):
    delta = 1e-8
    N = len(u)
    F0 = f_res(u, c, H, k)
    J = np.zeros((N+1, N+1))
    # ∂F/∂u_i
    for i in range(N):
        du = np.zeros(N)
        du[i] = delta
        J[:, i] = (f_res(u+du, c, H, k) - F0) / delta
    # ∂F/∂c
    J[:, -1] = (f_res(u, c+delta, H, k) - F0) / delta
    return J


def newton(H=0.5, N=128, L=np.pi, tol=1e-12, maxit=30):
    dx = 2*L/N
    x = np.linspace(-L, L, N, endpoint=False)
    k = fftfreq(N, d=dx) * 2*np.pi

    # Initial guess
    u = H/2 * np.cos(x)
    c = 1.0

    for _ in range(maxit):
        R = f_res(u, c, H, k)
        if np.linalg.norm(R, np.inf) < tol:
            break
        delta = np.linalg.solve(jacobian(u, c, H, k), -R)
        u += delta[:-1]
        c += delta[-1]

    return u, c, k

# Linear operator L


def L_matrix(U, c, k):
    N = len(U)
    Ux, _ = spatial_derivatives(U, k)
    L = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1
        ux_e, uxx_e = spatial_derivatives(e, k)
        L[:, j] = (
            -c*ux_e
            + hilbert(uxx_e, k)
            + Ux*e + U*ux_e
        )
    return L

# RHS and RK4 step


def rhs(u, c, k):
    ux, uxx = spatial_derivatives(u, k)
    return -c*hilbert(ux, k) - uxx + hilbert(u*ux, k)


def rk4_step(u, dt, rhs, *args):
    k1 = rhs(u, *args)
    k2 = rhs(u + 0.5*dt*k1, *args)
    k3 = rhs(u + 0.5*dt*k2, *args)
    k4 = rhs(u + dt*k3, *args)
    return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# 1) Compute base state
H = 0.5
U, c, k = newton(H=H, N=128, L=np.pi)

Lmat = L_matrix(U, c, k)
eigvals, eigvecs = np.linalg.eig(Lmat)


mode_inds = np.where(np.abs(np.imag(eigvals)) > 1e-8)[0]
mode_inds = mode_inds[np.argsort(np.abs(np.imag(eigvals[mode_inds])))]


eps = 1e-2
dt = 1e-4

omega_list, T_pred_list, T_meas_list = [], [], []

for idx in mode_inds:
    ω = np.imag(eigvals[idx])

    # ---- fast exit if |ω| ≥ 200  ---------------------------
    if abs(ω) >= 500:
        continue
    # --------------------------------------------------------

    V = eigvecs[:, idx]
    T_pred = 2 * np.pi / abs(ω)

    # Initial perturbation
    u = U + eps * (V + np.conj(V))

    # Time window: three periods (capped by max_steps)
    total_time = 3 * T_pred
    n_steps = int(total_time / dt)

    # Non-linear time-march + peak picking
    a = np.empty(n_steps + 1)
    t = np.linspace(0, total_time, n_steps + 1)
    for n in range(n_steps + 1):
        a[n] = np.dot(u - U, np.imag(V)) / np.linalg.norm(np.imag(V))**2
        if n < n_steps:
            u = rk4_step(u, dt, rhs, c, k)

    peaks = np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1
    T_meas = np.mean(np.diff(t[peaks])) if peaks.size >= 2 else np.nan

    omega_list.append(abs(ω))
    T_pred_list.append(T_pred)
    T_meas_list.append(T_meas)


# 5) Plot theory vs measured
plt.figure()
plt.plot(omega_list, T_pred_list, 'o-', label='Theory: $2π/ω$')
plt.plot(omega_list, T_meas_list, 'x-', label='Measured')
plt.xlabel('ω')
plt.ylabel('Period T')
plt.title('Predicted vs Measured Periods for Each Mode')
plt.legend()
plt.show()
