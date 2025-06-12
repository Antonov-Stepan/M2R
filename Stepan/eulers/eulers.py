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


N = 256
L = np.pi
T = 0.02
dt = 0.0001
steps = int(T / dt)

x = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
k_sign = np.sign(k)

u = 0.1 * np.cos(x)

snapshot_times = np.linspace(0, T, 100)
snapshot_steps = [int(t / dt) for t in snapshot_times]
snapshots = []
times = []

for step in range(steps + 1):
    if step in snapshot_steps:
        snapshots.append(u.copy())
        times.append(step * dt)
    u = u + dt * f(u, k, k_sign)

"""
# === Plotting and animation ===
fig, ax = plt.subplots()
line, = ax.plot(x, snapshots[0])
title = ax.set_title("t = 0.00")
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")

def update(frame):
    line.set_ydata(snapshots[frame])
    title.set_text(f"t = {times[frame]:.2f}")
    return line, title

ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots), interval=100, blit=False
)

plt.show()
"""

integral = []
for snapshot in snapshots:
    current = 0
    for step_val in snapshot:
        current += step_val * dx
    integral += [current]

integral2 = []
for snapshot in snapshots:
    current2 = 0
    for step_val in snapshot:
        current2 += (step_val ** 2) * dx
    integral2 += [current2]

plt.plot(snapshot_times, integral, label='Integral u', color='blue')
plt.plot(snapshot_times, integral2, label='Integral u^2', color='red')
plt.xlabel('X-axis')
plt.ylabel('U(x)')
plt.title('U(x) at t_n')
plt.legend()
plt.show()