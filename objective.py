"""
Objective function for the IRS beamforming optimization.

Given the phase shift vector θ, the transmit beamforming uses MRT
(Maximum Ratio Transmission), and the objective is:

    max  ||v^H Φ + h_d^H||²    (channel gain, proportional to rate)

The achievable rate (spectrum efficiency) is:
    R = log2(1 + P_T * ||v^H Φ + h_d^H||² / σ²)

Reference: Eq. (2), (6)-(10) in the paper.
"""

import numpy as np
from phase_shift_model import reflection_vector
from config import PT, SIGMA2


def compute_channel_gain(theta_vec, Phi, h_d, use_practical=True):
    """
    Compute the effective channel gain ||v^H Φ + h_d^H||².

    This is the inner objective that the optimization algorithms maximize.

    Parameters
    ----------
    theta_vec : ndarray, shape (N,) or (pop_size, N)
        Phase shift vector(s). If 2D, evaluates a batch (for PSO/CMA-ES).
    Phi : ndarray, shape (N, M)
        Combined channel matrix Φ = diag(h_r^H) G.
    h_d : ndarray, shape (M,)
        AP-to-User direct channel.
    use_practical : bool
        Whether to use the practical phase shift model.

    Returns
    -------
    float or ndarray
        Channel gain value(s).
    """
    if theta_vec.ndim == 1:
        # Single evaluation
        v = reflection_vector(theta_vec, use_practical)
        combined = v.conj() @ Phi + h_d.conj()     # shape (M,)
        return np.sum(np.abs(combined) ** 2)
    else:
        # Batch evaluation: theta_vec is (pop_size, N)
        v = reflection_vector(theta_vec, use_practical)      # (pop_size, N)
        combined = v.conj() @ Phi + h_d.conj()[np.newaxis, :]  # (pop_size, M)
        return np.sum(np.abs(combined) ** 2, axis=1)            # (pop_size,)


def compute_rate(channel_gain, pt=PT, sigma2=SIGMA2):
    """
    Compute the achievable rate (spectrum efficiency) in bps/Hz.

    R = log2(1 + P_T * channel_gain / σ²)

    Parameters
    ----------
    channel_gain : float or ndarray
        The channel gain ||v^H Φ + h_d^H||².
    pt : float
        Transmit power (Watts).
    sigma2 : float
        Noise variance (Watts).

    Returns
    -------
    float or ndarray
        Achievable rate in bps/Hz.
    """
    snr = pt * channel_gain / sigma2
    return np.log2(1 + snr)


def compute_lower_bound_rate(h_d, pt=PT, sigma2=SIGMA2):
    """
    Compute the achievable rate without IRS (lower bound).

    The optimal beamforming without IRS is MRT on h_d:
        w* = sqrt(P_T) * h_d / ||h_d||

    Achievable rate = log2(1 + P_T * ||h_d||² / σ²)

    Parameters
    ----------
    h_d : ndarray, shape (M,)
        AP-to-User direct channel.

    Returns
    -------
    float
        Lower bound rate in bps/Hz.
    """
    gain = np.sum(np.abs(h_d) ** 2)
    return compute_rate(gain, pt, sigma2)
