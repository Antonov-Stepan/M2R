"""Calculates the solution to B-O using RK4 method.

rk4.py

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


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

def rk4_step(u, dt, k):
    k1 = f(u, k, k_sign)
    k2 = f(u + 0.5 * dt * k1, k, k_sign)
    k3 = f(u + 0.5 * dt * k2, k, k_sign)
    k4 = f(u + dt * k3, k, k_sign)

    return u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Replace with your actual file path
df = pd.read_csv('C:/Users/2004s/OneDrive/Desktop/GProject/M2R/Kevin/solutions.csv')

first_value = df['u'].iloc[0].split(" ")


N = 256
L = np.pi
T = 5.0
dt = 0.0001
steps = int(T / dt)

x = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
k_sign = np.sign(k)

u = np.cos(x)

snapshot_times = np.linspace(0, T, 100)  # t = 0, 0.5, ..., 5.0
snapshot_steps = [int(t / dt) for t in snapshot_times]
snapshots = [u.copy()]
times = [0.0]

for step in range(1, steps + 1):

    u = rk4_step(u, dt, k)

    if step in snapshot_steps:
        snapshots.append(u.copy())
        times.append(step * dt)

"""
fig, ax = plt.subplots()
line, = ax.plot(x, snapshots[0])
ax.set_ylim(-2, 2)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
title = ax.set_title("t = 0.00")


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

