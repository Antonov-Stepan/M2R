import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import fsolve

# Parameters configuration
L = 100.0  # Half-width of spatial domain
N = 256  # Number of spatial points
c = 0.2  # Wave speed
max_iter = 100  # Maximum Newton iterations
tol = 1e-8  # Convergence tolerance

# Create spatial grid
x = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]

# Wave number vector (FFT frequencies)
k = 2 * np.pi * fftfreq(N, dx)


# Spectral operator for Hilbert transform
def hilbert_transform(u):
    u_hat = fft(u)
    # Hilbert transform: multiply by -i * sign(k) in spectral space
    H_u_hat = -1j * np.sign(k) * u_hat
    # Handle k=0 case
    H_u_hat[k == 0] = 0
    return np.real(ifft(H_u_hat))


# Spectral operator for second derivative
def second_derivative(u):
    u_hat = fft(u)
    u_xx_hat = -(k**2) * u_hat
    return np.real(ifft(u_xx_hat))


# Residual function (steady-state form of BO equation)
def residual(u):
    ux = np.gradient(u, dx)  # First derivative
    H_uxx = hilbert_transform(
        second_derivative(u))  # Hilbert transform of second derivative
    return -c * ux + u * ux + H_uxx


# Newton's method for solving traveling wave solutions
def newton_method(u_initial):
    u = u_initial.copy()
    errors = []

    for i in range(max_iter):
        # Compute current residual
        F = residual(u)
        error = np.linalg.norm(F)
        errors.append(error)

        if error < tol:
            print(f"Newton's method converged after {i+1} iterations")
            break

        # Compute Jacobian approximation (finite differences)
        J = np.zeros((N, N))
        epsilon = 1e-6

        for j in range(N):
            u_perturbed = u.copy()
            u_perturbed[j] += epsilon
            F_perturbed = residual(u_perturbed)
            J[:, j] = (F_perturbed - F) / epsilon

        # Solve linear system: J * du = -F
        try:
            du = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            du = np.linalg.pinv(J) @ (-F)

        # Update solution
        u += du

        # Apply amplitude constraint (maintain constant max-min difference)
        current_amp = np.max(u) - np.min(u)
        if i == 0:
            initial_amp = current_amp
        else:
            u = (u - np.min(u)) * (initial_amp / current_amp) + np.min(u)

    else:
        print(f"Newton's method did not converge after {max_iter} iterations")

    return u, errors


# Initial guess (single soliton solution)
def initial_guess(x, c):
    return 4 * c / (c**2 * x**2 + 1)


# Main execution
if __name__ == "__main__":
    # Initial guess
    u_initial = initial_guess(x, c)

    # Solve using Newton's method
    u_solution, errors = newton_method(u_initial)

    # Exact solution for comparison
    u_exact = initial_guess(x, c)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Solution comparison
    plt.subplot(2, 2, 1)
    plt.plot(x, u_initial, 'b--', label='Initial guess')
    plt.plot(x, u_solution, 'r-', label='Numerical solution')
    plt.plot(x, u_exact, 'g:', label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Traveling Wave Solution of Benjamin-Ono Equation')
    plt.legend()
    plt.grid(True)

    # Error convergence
    plt.subplot(2, 2, 2)
    plt.semilogy(range(len(errors)), errors, 'bo-')
    plt.xlabel('Iteration count')
    plt.ylabel('Residual norm')
    plt.title('Convergence History of Newton Method')
    plt.grid(True)

    # Difference between numerical and exact solutions
    plt.subplot(2, 2, 3)
    plt.plot(x, u_solution - u_exact, 'm-')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Difference: Numerical vs Exact Solution')
    plt.grid(True)

    # Residual distribution
    plt.subplot(2, 2, 4)
    residual_vals = residual(u_solution)
    plt.plot(x, residual_vals, 'c-')
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('Residual Distribution of Final Solution')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('benjamin_ono_traveling_wave.png')
    plt.show()

    # Calculate and print amplitude
    amplitude = np.max(u_solution) - np.min(u_solution)
    print(f"Solution amplitude (max - min): {amplitude:.6f}")

    # Save results
    np.savez('traveling_wave_solution.npz',
             x=x,
             u_initial=u_initial,
             u_solution=u_solution,
             u_exact=u_exact,
             errors=errors)
