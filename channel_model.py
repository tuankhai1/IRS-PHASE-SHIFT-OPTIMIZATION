"""
Channel model for IRS-aided MISO wireless communication system.

Generates fading channels with distance-dependent path loss
for the AP-User, IRS-User, and AP-IRS links.

Channel models:
    - AP→User:  Rayleigh fading (NLoS, α=3.8)
    - IRS→User: Rayleigh fading (NLoS, α=2.8)
    - AP→IRS:   Rician fading   (LoS,  α=2.2, K_rice from config)

Geometry:
    AP at (0, 0)
    IRS at (D_AP_IRS, 0)
    User at (d_horizontal, D_VERTICAL)
"""

import numpy as np
from config import (
    M, D_AP_IRS, D_VERTICAL, C0,
    ALPHA_AI, ALPHA_IU, ALPHA_AU,
    K_RICIAN_AI
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


def _generate_rician(shape, path_loss, K_rice, rng):
    """
    Generate a Rician fading channel matrix.

    h = sqrt(PL) * [ sqrt(K/(K+1)) * h_LoS  +  sqrt(1/(K+1)) * h_NLoS ]

    The LoS component uses a deterministic uniform-phase steering vector
    (random phase per element, fixed magnitude) which models the dominant
    specular path without requiring explicit array geometry.

    Parameters
    ----------
    shape : tuple
        Output shape (e.g. (N, M)).
    path_loss : float
        Large-scale path loss coefficient.
    K_rice : float
        Rician K-factor in linear scale.
    rng : np.random.Generator

    Returns
    -------
    ndarray, complex
        Channel matrix of the given shape.
    """
    # LoS component: unit-magnitude with random phase (specular)
    h_los = np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))

    # NLoS component: Rayleigh ~ CN(0, 1)
    h_nlos = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    # Combine with Rician K-factor weighting
    h = (np.sqrt(K_rice / (K_rice + 1)) * h_los +
         np.sqrt(1.0 / (K_rice + 1)) * h_nlos)

    return np.sqrt(path_loss) * h


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

    # AP→User: Rayleigh fading (NLoS, severe path loss)
    h_d = np.sqrt(pl_au / 2) * (rng.standard_normal(M) + 1j * rng.standard_normal(M))

    # IRS→User: Rayleigh fading (NLoS)
    h_r = np.sqrt(pl_iu / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    # AP→IRS: Rician fading (strong LoS, low path-loss exponent α=2.2)
    G = _generate_rician((N, M), pl_ai, K_RICIAN_AI, rng)

    # Combined channel matrix: Φ = diag(h_r^H) @ G
    Phi = np.diag(h_r.conj()) @ G

    return h_d, Phi
