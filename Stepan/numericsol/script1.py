import numpy as np
import matplotlib.pyplot as plt

# Sample parameters
L = 2 * np.pi          # interval length
N = 512                # number of sample points
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]

# Define the function
f = np.sin(5*x) + 0.5 * np.cos(10*x)

# Compute FFT
F = np.fft.fft(f)
freqs = np.fft.fftfreq(N, d=dx)

# Normalize amplitude
amplitudes = np.abs(F) / N

# Plot only positive frequencies
half = N // 2
plt.figure(figsize=(12, 5))
plt.stem(freqs[:half], amplitudes[:half])
plt.title("Frequency Spectrum of f(x) = sin(5x) + 0.5*cos(10x)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
