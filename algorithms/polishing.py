"""
Shared gradient polishing utility for IRS phase shift optimization.

Provides analytical gradient ascent refinement that can be used by
any metaheuristic (PSO, CMA-ES, etc.) to polish a candidate solution.

Unlike AO's coordinate descent (one element at a time), gradient
ascent updates ALL elements simultaneously in the direction of
steepest increase. This can escape coordinate-wise local optima
where no single-element improvement exists but a multi-element step
would improve the objective.
"""

import numpy as np
from phase_shift_model import wrap_angle, beta, reflection_vector
from objective import compute_channel_gain
from config import BETA_MIN, K_PARAM, PHI_PARAM


def gradient_polish(theta, Phi, h_d, use_practical, n_steps=20, lr_init=0.05):
    """
    Refine a phase shift solution using analytical gradient ascent.

    The gradient is computed analytically from the channel model:
        ∂f/∂θ_n = 2 Re(∂v_n*/∂θ_n · q_n)
    where q = Φ (v^H Φ + h_d^H)^H is the back-projected effective channel.

    Parameters
    ----------
    theta : ndarray, shape (N,)
        Starting phase shift vector.
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct channel.
    use_practical : bool
        Whether to use practical phase shift model.
    n_steps : int
        Maximum number of gradient ascent steps.
    lr_init : float
        Initial learning rate.

    Returns
    -------
    best_theta : ndarray, shape (N,)
        Refined phase shift vector.
    best_obj : float
        Refined channel gain.
    """
    theta = wrap_angle(theta.copy())
    best_obj = compute_channel_gain(theta, Phi, h_d, use_practical)
    best_theta = theta.copy()
    lr = lr_init

    for _ in range(n_steps):
        # Compute effective channel and back-projection
        v = reflection_vector(theta, use_practical)
        h_eff = v.conj() @ Phi + h_d.conj()          # (M,)
        q = Phi @ h_eff.conj()                       # (N,)

        # Analytical gradient
        if use_practical:
            b = beta(theta)
            s = (np.sin(theta - PHI_PARAM) + 1) / 2
            b_prime = ((1 - BETA_MIN) * K_PARAM *
                       np.maximum(s, 1e-20) ** (K_PARAM - 1) *
                       np.cos(theta - PHI_PARAM) / 2)
            dv_conj = (b_prime - 1j * b) * np.exp(-1j * theta)
        else:
            dv_conj = -1j * np.exp(-1j * theta)

        grad = 2.0 * np.real(dv_conj * q)

        # Gradient ascent with adaptive step size
        theta_new = wrap_angle(theta + lr * grad)
        obj_new = compute_channel_gain(theta_new, Phi, h_d, use_practical)

        if obj_new > best_obj:
            best_obj = obj_new
            best_theta = theta_new.copy()
            theta = theta_new
            lr = min(lr * 1.1, 0.5)
        else:
            lr *= 0.5
            theta = best_theta.copy()
            if lr < 1e-8:
                break

    return best_theta, best_obj
