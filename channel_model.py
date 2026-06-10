"""
Channel model for IRS-aided MISO wireless communication system.

Generates Rayleigh fading channels with distance-dependent path loss
for the AP-User, IRS-User, and AP-IRS links.

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


def generate_channels(N, d_horizontal, rng=None):
    """
    Generate one realization of Rayleigh fading channels with path loss.

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
        AP-to-User direct channel (complex).
    h_r : ndarray, shape (N,)
        IRS-to-User channel (complex).
    G : ndarray, shape (N, M)
        AP-to-IRS channel (complex).
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

    # Rayleigh fading: each element ~ CN(0, PL)
    # CN(0, σ²) = N(0, σ²/2) + j*N(0, σ²/2)
    h_d = np.sqrt(pl_au / 2) * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    h_r = np.sqrt(pl_iu / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    G = np.sqrt(pl_ai / 2) * (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M)))

    # Combined channel matrix: Φ = diag(h_r^H) @ G
    Phi = np.diag(h_r.conj()) @ G

    return h_d, h_r, G, Phi
