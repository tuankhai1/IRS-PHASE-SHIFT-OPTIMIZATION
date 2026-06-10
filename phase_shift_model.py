"""
Practical Phase Shift Model for IRS reflecting elements.

The key insight is that the reflection amplitude depends on the phase shift:
    β(θ) = (1 - β_min) * ((sin(θ - φ) + 1) / 2)^k + β_min

This model captures the physics of real IRS hardware where energy
dissipation varies with phase shift (minimum amplitude near θ=0,
maximum near θ=±π).

Reference: Eq. (5) in the paper.
"""

import numpy as np
from config import BETA_MIN, K_PARAM, PHI_PARAM


def beta(theta, beta_min=BETA_MIN, k=K_PARAM, phi=PHI_PARAM):
    """
    Compute the reflection amplitude for given phase shifts
    using the practical phase shift model.

    β(θ) = (1 - β_min) * ((sin(θ - φ) + 1) / 2)^k + β_min

    Parameters
    ----------
    theta : float or ndarray
        Phase shift(s) in radians, in [-π, π].
    beta_min : float
        Minimum reflection amplitude.
    k : float
        Steepness parameter.
    phi : float
        Phase offset parameter.

    Returns
    -------
    float or ndarray
        Reflection amplitude(s) in [β_min, 1].
    """
    return (1 - beta_min) * ((np.sin(theta - phi) + 1) / 2) ** k + beta_min


def reflection_vector(theta_vec, use_practical=True):
    """
    Compute the IRS reflection coefficient vector.

    For practical model:  v_n = β(θ_n) * e^{jθ_n}
    For ideal model:      v_n = e^{jθ_n}   (|v_n| = 1)

    Parameters
    ----------
    theta_vec : ndarray, shape (N,)
        Phase shifts for each reflecting element.
    use_practical : bool
        If True, use practical model. If False, use ideal model (β=1).

    Returns
    -------
    v : ndarray, shape (N,), complex
        Reflection coefficient vector.
    """
    if use_practical:
        amplitudes = beta(theta_vec)
    else:
        amplitudes = np.ones_like(theta_vec)
    return amplitudes * np.exp(1j * theta_vec)


def wrap_angle(theta):
    """
    Wrap angle(s) to [-π, π).

    Parameters
    ----------
    theta : float or ndarray
        Angle(s) in radians.

    Returns
    -------
    float or ndarray
        Wrapped angle(s).
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi
