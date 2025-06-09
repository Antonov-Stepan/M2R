import numpy as np


def hilbert_transform(u, k_sign):
    """Compute Hilbert transform via Fourier spectral method.
    
    Args:
        u (ndarray): Input wave function
        k_sign (ndarray): Sign vector of wavenumbers
    
    Returns:
        ndarray: Hilbert-transformed result (real part)
    """
    u_hat = np.fft.fft(u)  # Fourier transform
    H_hat = -1j * k_sign * u_hat  # Apply Hilbert operator in Fourier space
    return np.fft.ifft(H_hat).real  # Inverse Fourier transform


def spatial_derivatives(u, k):
    """Compute 1st and 2nd spatial derivatives in Fourier domain.
    
    Args:
        u (ndarray): Wave function values
        k (ndarray): Wavenumber vector
    
    Returns:
        tuple: (ux, uxx) - 1st and 2nd derivatives
    """
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real  # 1st derivative (real part)
    uxx = np.fft.ifft(-k**2 * u_hat).real  # 2nd derivative (real part)
    return ux, uxx


def f2(u, c, amplitude):
    """Evaluate residual of traveling wave equation with symmetry constraint.
    
    Args:
        u (ndarray): Half-domain wave solution
        c (float): Wave speed
        amplitude (float): Target wave amplitude
    
    Returns:
        ndarray: Residual vector + constraint
    """
    # Reflect half-domain to full domain
    uf = np.concatenate((u, u[-2:0:-1]))

    # Spatial grid setup
    L = np.pi
    N = (len(u) - 1) * 2
    X = np.linspace(-L, L, num=N, endpoint=False)
    dx = X[1] - X[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_sign = np.sign(k)

    # Compute derivatives and Hilbert transforms
    ux, uxx = spatial_derivatives(uf, k)
    H_ux = hilbert_transform(ux, k_sign)
    H_uux = hilbert_transform(uf * ux, k_sign)

    # Benjamin-Ono equation residual
    fu = (-c * H_ux) - uxx + H_uux
    fu = fu[0:(N // 2 + 1)]  # Restrict to half-domain

    # Amplitude constraint: u(max) - u(0) = amplitude
    constraint = u[-1] - u[0] - amplitude
    return np.append(fu, constraint)


def jacobian2(u, c, amplitude, res):
    """Compute Jacobian matrix via finite differences.
    
    Args:
        u (ndarray): Current solution vector
        c (float): Wave speed
        amplitude (float): Target amplitude
        res (ndarray): Precomputed residual from f2()
    
    Returns:
        ndarray: (N+1 x N+1) Jacobian matrix
    """
    N = len(u)
    delta = 1e-7  # Finite difference step
    J = np.zeros((N + 1, N + 1))

    # Derivatives w.r.t u (wave function)
    for j in range(N):
        delta_uj = u.copy()  # Avoid mutation
        delta_uj[j] += delta  # Perturb j-th component
        delta_res = f2(delta_uj, c, amplitude)
        J[:, j] = (delta_res - res) / delta  # Finite difference

    # Derivatives w.r.t c (wave speed)
    delta_c_res = f2(u, c + delta, amplitude)
    J[:, -1] = (delta_c_res - res) / delta

    return J


def newton_meth2(N, u, amp, c=0.0):
    """Solve traveling wave equation using Newton's method with half-domain symmetry.
    
    Args:
        N (int): Total grid points (full domain)
        u (ndarray): Initial guess (half-domain)
        amp (float): Target wave amplitude
        c (float): Initial wave speed guess
    
    Returns:
        tuple: (uf, c) - Full-domain solution and optimized wave speed
    """
    Nh = N // 2 + 1  # Half-domain size
    ui = u[0:Nh].copy()  # Enforce half-domain
    res = f2(ui, c, amp)
    err = np.max(np.abs(res))

    # Newton iteration loop
    while err > 1e-10:
        J = jacobian2(ui, c, amp, res)
        corr = np.linalg.solve(-J, res)  # Solve linear system

        # Update solution and wave speed
        ui += corr[:-1]
        c += corr[-1]

        # Recompute residual
        res = f2(ui, c, amp)
        err = np.max(np.abs(res))

    # Reconstruct full symmetric solution
    uf = np.concatenate((ui, ui[-2:0:-1]))
    return uf, c
