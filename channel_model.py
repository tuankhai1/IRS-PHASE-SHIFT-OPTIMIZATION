"""
Channel model for IRS-aided MISO wireless communication system.

Generates fading channels with distance-dependent path loss
for the AP-User, IRS-User, and AP-IRS links.

Channel models:
    - AP→User:  Rayleigh fading (α=3.8)
    - IRS→User: Rayleigh fading (α=2.8)
    - AP→IRS:   Rayleigh fading (α=2.2)

Geometry:
    AP at (0, 0)
    IRS at (D_AP_IRS, 0)
    User at (d_horizontal, D_VERTICAL)
"""

import numpy as np
from config import (
    M, D_AP_IRS, D_VERTICAL, C0,
    ALPHA_AI, ALPHA_IU, ALPHA_AU
)


def compute_path_loss(distance, exponent):
    """
    Compute path loss at a given distance.

    PL(d) = C0 * d^(-alpha)

    Parameters
    ----------
    distance : float
        Distance in meters (must be > 0).
    exponent : float
        Path loss exponent.

    Returns
    -------
    float
        Path loss coefficient (linear scale, < 1).
    """
    return C0 * distance ** (-exponent)


def compute_distances(d_horizontal):
    """
    Compute the three link distances given the user's horizontal position.

    Parameters
    ----------
    d_horizontal : float
        Horizontal distance from the AP to the user (meters).

    Returns
    -------
    d_au : float
        AP-to-User distance.
    d_iu : float
        IRS-to-User distance.
    d_ai : float
        AP-to-IRS distance (fixed at D_AP_IRS).
    """
    d_au = np.sqrt(d_horizontal ** 2 + D_VERTICAL ** 2)
    d_iu = np.sqrt((D_AP_IRS - d_horizontal) ** 2 + D_VERTICAL ** 2)
    d_ai = D_AP_IRS
    return d_au, d_iu, d_ai


def _generate_rayleigh(shape, path_loss, rng):
    """Generate a Rayleigh fading channel with distance-dependent path loss."""
    return np.sqrt(path_loss / 2) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def generate_channels(N, d_horizontal, rng=None):
    """
    Generate one realization of fading channels with path loss.

    Parameters
    ----------
    N : int
        Number of IRS reflecting elements.
    d_horizontal : float
        Horizontal distance from the AP to the user (meters).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    h_d : ndarray, shape (M,)
        AP-to-User direct channel (complex, Rayleigh).
    Phi : ndarray, shape (N, M)
        Combined channel matrix Φ = diag(h_r^H) @ G.
    """
    if rng is None:
        rng = np.random.default_rng()

    d_au, d_iu, d_ai = compute_distances(d_horizontal)

    # Path loss coefficients (large-scale fading)
    pl_au = compute_path_loss(d_au, ALPHA_AU)
    pl_iu = compute_path_loss(d_iu, ALPHA_IU)
    pl_ai = compute_path_loss(d_ai, ALPHA_AI)

    # AP→User: Rayleigh fading
    h_d = _generate_rayleigh(M, pl_au, rng)

    # IRS→User: Rayleigh fading
    h_r = _generate_rayleigh(N, pl_iu, rng)

    # AP→IRS: Rayleigh fading
    G = _generate_rayleigh((N, M), pl_ai, rng)

    # Combined channel matrix: Φ = diag(h_r^H) @ G
    Phi = np.diag(h_r.conj()) @ G

    return h_d, Phi
