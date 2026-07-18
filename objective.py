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
from circuit_model import components_to_reflection_vector
from config import PT, SIGMA2, OMEGA, Z0


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
        assert theta_vec.shape[0] == Phi.shape[0], (
            f"theta_vec length {theta_vec.shape[0]} != N={Phi.shape[0]}")
        v = reflection_vector(theta_vec, use_practical)
        combined = v.conj() @ Phi + h_d.conj()     # shape (M,)
        return np.sum(np.abs(combined) ** 2)
    else:
        # Batch evaluation: theta_vec is (pop_size, N)
        assert theta_vec.shape[1] == Phi.shape[0], (
            f"theta_vec dim-1 {theta_vec.shape[1]} != N={Phi.shape[0]}")
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


def compute_channel_gain_from_components(x, N, Phi, h_d,
                                         omega=OMEGA, Z0_val=Z0):
    """
    Compute channel gain from component parameters.

    Steps 3-5 of the general procedure:
        3. Compute Z_n from L1, L2, C, R
        4. Compute v_n = (Z_n - Z0) / (Z_n + Z0)
        5. Form v, compute ||v^H Φ + h_d^H||²

    Parameters
    ----------
    x : ndarray, shape (4*N,) or (pop_size, 4*N)
        Component parameters [L1_1, L2_1, C_1, R_1, ..., L1_N, L2_N, C_N, R_N].
    N : int
        Number of IRS reflecting elements.
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct AP-to-User channel.
    omega : float
        Angular frequency (rad/s).
    Z0_val : float
        Free-space impedance (Ω).

    Returns
    -------
    float or ndarray
        Channel gain value(s).
    """
    v = components_to_reflection_vector(x, N, omega, Z0_val)

    if v.ndim == 1:
        combined = v.conj() @ Phi + h_d.conj()         # (M,)
        return np.sum(np.abs(combined) ** 2)
    else:
        combined = v.conj() @ Phi + h_d.conj()[np.newaxis, :]  # (pop, M)
        return np.sum(np.abs(combined) ** 2, axis=1)           # (pop,)


def compute_rate_from_components(x, N, Phi, h_d,
                                 omega=OMEGA, Z0_val=Z0,
                                 pt=PT, sigma2=SIGMA2):
    """
    Compute achievable rate R_SE from component parameters.

    This is the fitness function used by component-level optimizers
    (Step 6 of the general procedure).

    R_SE = log2(1 + P_T · ||v^H Φ + h_d^H||² / σ²)

    Parameters
    ----------
    x : ndarray, shape (4*N,) or (pop_size, 4*N)
        Component parameters.
    N : int
        Number of IRS reflecting elements.
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct AP-to-User channel.
    omega : float
        Angular frequency (rad/s).
    Z0_val : float
        Free-space impedance (Ω).
    pt : float
        Transmit power (Watts).
    sigma2 : float
        Noise variance (Watts).

    Returns
    -------
    float or ndarray
        Achievable rate in bps/Hz.
    """
    gain = compute_channel_gain_from_components(x, N, Phi, h_d, omega, Z0_val)
    return compute_rate(gain, pt, sigma2)
