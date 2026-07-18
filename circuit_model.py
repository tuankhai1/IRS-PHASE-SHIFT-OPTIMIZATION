"""
Circuit impedance model for IRS reflecting elements.

Models each IRS element as a parallel resonant circuit with parameters:
    L1 — coupling inductance
    L2 — varactor series inductance
    C  — varactor capacitance (tunable)
    R  — varactor equivalent series resistance (ESR)

The equivalent impedance of the n-th IRS element is (Eq. 3):
    Z_n = jωL₁(jωL₂ + 1/(jωC) + R) / (jωL₁ + jωL₂ + 1/(jωC) + R)

The reflection coefficient is (Eq. 4):
    v_n = (Z_n - Z₀) / (Z_n + Z₀)

Operating frequency: 5.8 GHz (ISM band).
    Ref: ITU Radio Regulations Art. 5, Footnote 5.150;
         C. Liaskos et al., "A New Wireless Communication Paradigm
         through Software-Controlled Metasurfaces," IEEE Commun. Mag., 2018.

Component ranges based on typical varactor-based IRS element designs:
    Ref: Skyworks SMV1231-079LF datasheet; extended for broader coverage.
"""

import numpy as np
from config import OMEGA, Z0, L1_BOUNDS, L2_BOUNDS, C_BOUNDS, R_BOUNDS


def compute_impedance(L1, L2, C, R, omega=OMEGA):
    """
    Compute the equivalent impedance Z_n of an IRS element.

    Z_n = jωL₁ · (jωL₂ + 1/(jωC) + R) / (jωL₁ + jωL₂ + 1/(jωC) + R)

    Parameters
    ----------
    L1 : float or ndarray
        Coupling inductance (H).
    L2 : float or ndarray
        Varactor series inductance (H).
    C : float or ndarray
        Varactor capacitance (F).
    R : float or ndarray
        Varactor series resistance (Ω).
    omega : float
        Angular frequency (rad/s).

    Returns
    -------
    complex or ndarray of complex
        Equivalent impedance Z_n.
    """
    jw = 1j * omega
    Z_L1 = jw * L1                               # jωL₁
    Z_series = jw * L2 + 1.0 / (jw * C) + R     # jωL₂ + 1/(jωC) + R
    return (Z_L1 * Z_series) / (Z_L1 + Z_series)


def compute_reflection_coefficient(Z, Z0_val=Z0):
    """
    Compute the reflection coefficient from impedance.

    v_n = (Z_n - Z₀) / (Z_n + Z₀)

    Parameters
    ----------
    Z : complex or ndarray of complex
        Impedance value(s).
    Z0_val : float
        Free-space impedance (default: 377 Ω).

    Returns
    -------
    complex or ndarray of complex
        Reflection coefficient v_n.
    """
    return (Z - Z0_val) / (Z + Z0_val)


def components_to_reflection_vector(x, N, omega=OMEGA, Z0_val=Z0):
    """
    Convert component parameter vector(s) to IRS reflection vector(s).

    End-to-end: extracts L1, L2, C, R → computes Z_n → computes v_n.

    Parameters
    ----------
    x : ndarray, shape (4*N,) or (pop_size, 4*N)
        Component parameters ordered as:
        [L1_1, L2_1, C_1, R_1, L1_2, L2_2, C_2, R_2, ...,
         L1_N, L2_N, C_N, R_N]
    N : int
        Number of IRS reflecting elements.
    omega : float
        Angular frequency (rad/s).
    Z0_val : float
        Free-space impedance (Ω).

    Returns
    -------
    v : ndarray, shape (N,) or (pop_size, N), complex
        Reflection coefficient vector(s).
    """
    if x.ndim == 1:
        # Single evaluation: x is (4*N,)
        params = x.reshape(N, 4)          # (N, 4)
        L1 = params[:, 0]
        L2 = params[:, 1]
        C  = params[:, 2]
        R  = params[:, 3]
    else:
        # Batch evaluation: x is (pop_size, 4*N)
        pop_size = x.shape[0]
        params = x.reshape(pop_size, N, 4)  # (pop_size, N, 4)
        L1 = params[:, :, 0]
        L2 = params[:, :, 1]
        C  = params[:, :, 2]
        R  = params[:, :, 3]

    Z = compute_impedance(L1, L2, C, R, omega)
    v = compute_reflection_coefficient(Z, Z0_val)
    return v


def get_component_bounds(N):
    """
    Get the lower and upper bounds for the 4*N component parameters.

    The bounds are tiled across all N elements.

    Parameters
    ----------
    N : int
        Number of IRS reflecting elements.

    Returns
    -------
    lower : ndarray, shape (4*N,)
        Lower bounds for each parameter.
    upper : ndarray, shape (4*N,)
        Upper bounds for each parameter.
    """
    lower_single = np.array([L1_BOUNDS[0], L2_BOUNDS[0],
                             C_BOUNDS[0], R_BOUNDS[0]])
    upper_single = np.array([L1_BOUNDS[1], L2_BOUNDS[1],
                             C_BOUNDS[1], R_BOUNDS[1]])
    return np.tile(lower_single, N), np.tile(upper_single, N)
