import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0
N = 2048
x = np.linspace(-L, L, N, endpoint=False)
h = x[1] - x[0]
k = 0.25
T = 5.0
steps = int(T / k)
q = 3  # Fixed-point iterations per time step

# Initial condition
A = 1.0
def initial_condition(x):
    return A / (1 + np.sin(np.pi * x / (2 * L))**2)

U = initial_condition(x)

# Frequency domain
p = np.fft.fftfreq(N, d=h)
k_vals = 2 * np.pi * p
sign_p = np.sign(p)
hilbert_laplacian = -1j * sign_p * (k_vals**2)

# Crank–Nicolson coefficients
C1_vals = 1 + 0.5 * k * hilbert_laplacian
C2_vals = 1 / (1 - 0.5 * k * hilbert_laplacian)

# Zabusky–Kruskal nonlinear term
def nonlinear_term(u):
    avg = (np.roll(u, -1) + u + np.roll(u, 1)) / 3
    deriv = (np.roll(u, -1) - np.roll(u, 1)) / (2 * h)
    return avg * deriv

# Store snapshots
snapshot_times = np.linspace(0, T, 11)  # t = 0, 0.5, ..., 5.0
snapshot_steps = [int(t / k) for t in snapshot_times]
snapshots = [U.copy()]
times = [0.0]

# Time-stepping loop
for step in range(1, steps + 1):
    U_hat_prev = np.fft.fft(U)
    g_hat = C1_vals * U_hat_prev
    W = U.copy()

    for _ in range(q):
        V = nonlinear_term(0.5 * (W + U))
        V_hat = np.fft.fft(V)
        W_hat = C2_vals * (g_hat - k * V_hat)
        W = np.fft.ifft(W_hat).real

    U = W.copy()

    if step in snapshot_steps:
        snapshots.append(U.copy())
        times.append(step * k)

# Parameters
c = 0.25        # wave speed (you can change)
t = 0.5        # fixed time (you can change)

# x values
x = np.linspace(-np.pi, np.pi, 2048)

# Function u(x,t)
u = 4 * c / (c**2 * (x - c * t)**2 + 1)
# Plot
# Plot snapshots
plt.figure(figsize=(12, 6))
for i, u_snapshot in enumerate(snapshots):
    plt.plot(x, u_snapshot, label=f"t = {times[i]:.1f}")
plt.plot(x, u, label=f't={t}, c={c}', linestyle='--')
plt.title("Time Evolution of the Benjamin–Ono Equation (Snapshots)")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
