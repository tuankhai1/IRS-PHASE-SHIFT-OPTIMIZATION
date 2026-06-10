"""
Numba JIT-compiled kernels for performance-critical inner loops.

These kernels eliminate Python interpreter overhead for the
Alternating Optimization (AO) algorithm's inner loop, which
iterates over N IRS elements and evaluates up to 1000 candidate
phase shifts per element.

Falls back gracefully if Numba is not installed.
"""

import numpy as np

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @numba.njit(cache=True)
    def ao_inner_iteration(theta, v, Psi, hd_hat, candidates, beta_cache,
                           beta_sq_cache, cos_cache, sin_cache,
                           N, n_cand, use_practical, beta_min, k_param, phi_param):
        """
        JIT-compiled AO inner iteration over all N elements (continuous case).

        Replaces the Python ``for n in range(N)`` loop with compiled code.
        Uses precomputed lookup tables for the 1D search.

        Parameters
        ----------
        theta : ndarray, shape (N,)
            Current phase shift vector (modified in-place).
        v : ndarray, shape (N,), complex
            Current reflection vector (modified in-place).
        Psi : ndarray, shape (N, N), complex
            Precomputed Φ Φ^H.
        hd_hat : ndarray, shape (N,), complex
            Precomputed Φ h_d.
        candidates : ndarray, shape (n_cand,)
            Precomputed candidate phase shift values.
        beta_cache : ndarray, shape (n_cand,)
            Precomputed β(candidates).
        beta_sq_cache : ndarray, shape (n_cand,)
            Precomputed β²(candidates).
        cos_cache : ndarray, shape (n_cand,)
            Precomputed cos(candidates).
        sin_cache : ndarray, shape (n_cand,)
            Precomputed sin(candidates).
        N : int
            Number of IRS elements.
        n_cand : int
            Number of candidate points.
        use_practical : bool
            Whether to use practical phase shift model.
        beta_min, k_param, phi_param : float
            Phase shift model parameters.

        Returns
        -------
        theta, v : modified in-place arrays.
        """
        for n in range(N):
            # Compute phi_n = 2 * (Psi[n,:] @ v - Psi[n,n]*v[n] + hd_hat[n])
            psi_dot_v = 0.0 + 0.0j
            for m in range(N):
                psi_dot_v += Psi[n, m] * v[m]
            psi_dot_v -= Psi[n, n] * v[n]
            phi_n = 2.0 * (psi_dot_v + hd_hat[n])

            psi_nn = Psi[n, n].real

            if not use_practical:
                # Ideal model: optimal theta is simply arg(phi_n)
                theta[n] = np.arctan2(phi_n.imag, phi_n.real)
                v[n] = np.cos(theta[n]) + 1j * np.sin(theta[n])
                continue

            # Practical model: search over precomputed candidates
            phi_n_abs = np.abs(phi_n)
            phi_n_arg = np.arctan2(phi_n.imag, phi_n.real)
            cos_arg = np.cos(phi_n_arg)
            sin_arg = np.sin(phi_n_arg)

            best_idx = 0
            best_val = -1e30
            for i in range(n_cand):
                cos_diff = cos_arg * cos_cache[i] + sin_arg * sin_cache[i]
                f_val = beta_sq_cache[i] * psi_nn + beta_cache[i] * phi_n_abs * cos_diff
                if f_val > best_val:
                    best_val = f_val
                    best_idx = i

            theta[n] = candidates[best_idx]

            # Update v[n] = beta(theta[n]) * exp(j*theta[n])
            s = (np.sin(theta[n] - phi_param) + 1.0) / 2.0
            b = (1.0 - beta_min) * s ** k_param + beta_min
            v[n] = b * (np.cos(theta[n]) + 1j * np.sin(theta[n]))

        return theta, v

    @numba.njit(cache=True)
    def ao_inner_iteration_discrete(theta, v, Psi, hd_hat, discrete_set,
                                    N, n_discrete, use_practical,
                                    beta_min, k_param, phi_param):
        """
        JIT-compiled AO inner iteration with discrete phase set.

        Parameters
        ----------
        theta : ndarray, shape (N,)
        v : ndarray, shape (N,), complex
        Psi : ndarray, shape (N, N), complex
        hd_hat : ndarray, shape (N,), complex
        discrete_set : ndarray, shape (n_discrete,)
        N, n_discrete : int
        use_practical : bool
        beta_min, k_param, phi_param : float

        Returns
        -------
        theta, v : modified in-place arrays.
        """
        for n in range(N):
            psi_dot_v = 0.0 + 0.0j
            for m in range(N):
                psi_dot_v += Psi[n, m] * v[m]
            psi_dot_v -= Psi[n, n] * v[n]
            phi_n = 2.0 * (psi_dot_v + hd_hat[n])

            psi_nn = Psi[n, n].real
            phi_n_abs = np.abs(phi_n)
            phi_n_arg = np.arctan2(phi_n.imag, phi_n.real)
            cos_arg = np.cos(phi_n_arg)
            sin_arg = np.sin(phi_n_arg)

            best_idx = 0
            best_val = -1e30
            for i in range(n_discrete):
                # Compute beta for this discrete value
                if use_practical:
                    s = (np.sin(discrete_set[i] - phi_param) + 1.0) / 2.0
                    b = (1.0 - beta_min) * s ** k_param + beta_min
                else:
                    b = 1.0
                cos_diff = cos_arg * np.cos(discrete_set[i]) + sin_arg * np.sin(discrete_set[i])
                f_val = b * b * psi_nn + b * phi_n_abs * cos_diff
                if f_val > best_val:
                    best_val = f_val
                    best_idx = i

            theta[n] = discrete_set[best_idx]

            # Update v[n]
            if use_practical:
                s = (np.sin(theta[n] - phi_param) + 1.0) / 2.0
                b = (1.0 - beta_min) * s ** k_param + beta_min
            else:
                b = 1.0
            v[n] = b * (np.cos(theta[n]) + 1j * np.sin(theta[n]))

        return theta, v

else:
    # Fallback stubs when Numba is not installed
    ao_inner_iteration = None
    ao_inner_iteration_discrete = None
