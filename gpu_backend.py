"""
GPU acceleration backend using CuPy for batch operations.

Provides GPU-accelerated batch channel gain computation used by
PSO and CMA-ES population-based optimizers.

Auto-detects CuPy availability and falls back to CPU (NumPy) seamlessly.
"""

import numpy as np
from config import BETA_MIN, K_PARAM, PHI_PARAM

import warnings

try:
    with warnings.catch_warnings():
        # Suppress the specific warning about CUDA_PATH not being found
        # so it doesn't spam the console if CuPy is installed but CUDA isn't.
        warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
        pass # cupy import removed
    # Forced to False to use CPU processing only
    GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False


class GPUAccelerator:
    """
    GPU-accelerated batch channel gain evaluator.

    Keeps channel matrices (Phi, h_d) on GPU memory for the duration
    of an optimization run, eliminating repeated CPU-to-GPU transfers.

    Usage::

        gpu = GPUAccelerator(Phi, h_d)
        gains = gpu.batch_channel_gain(theta_batch, use_practical)
        gpu.release()  # optional — frees GPU memory

    Falls back to CPU (NumPy) transparently if CuPy is not available.
    """

    def __init__(self, Phi, h_d):
        """
        Upload channel matrices to GPU (or store for CPU fallback).

        Parameters
        ----------
        Phi : ndarray, shape (N, M)
            Combined channel matrix.
        h_d : ndarray, shape (M,)
            Direct channel.
        """
        self._active = GPU_AVAILABLE
        if self._active:
            self._Phi = cp.asarray(Phi)
            self._h_d = cp.asarray(h_d)
        else:
            self._Phi = Phi
            self._h_d = h_d

    def release(self):
        """Free GPU memory."""
        self._Phi = None
        self._h_d = None

    def batch_channel_gain(self, theta_batch, use_practical):
        """
        Compute channel gain for a batch of phase shift vectors.

        Parameters
        ----------
        theta_batch : ndarray, shape (pop_size, N) or (N,)
            Phase shift vector(s).
        use_practical : bool
            Whether to use practical phase shift model.

        Returns
        -------
        float or ndarray
            Channel gain value(s). Always returned as NumPy scalar/array.
        """
        if self._active:
            return self._batch_gpu(theta_batch, use_practical)
        return self._batch_cpu(theta_batch, use_practical)

    def _batch_gpu(self, theta_batch, use_practical):
        """GPU path using CuPy."""
        try:
            theta_gpu = cp.asarray(theta_batch)

            # Compute reflection amplitudes on GPU
            if use_practical:
                amplitudes = ((1 - BETA_MIN) *
                              ((cp.sin(theta_gpu - PHI_PARAM) + 1) / 2) ** K_PARAM +
                              BETA_MIN)
            else:
                amplitudes = cp.ones_like(theta_gpu, dtype=cp.float64)

            v = amplitudes * cp.exp(1j * theta_gpu)

            # Compute channel gain: ||v^H Phi + h_d^H||^2
            if theta_gpu.ndim == 1:
                combined = v.conj() @ self._Phi + self._h_d.conj()
                gain = float(cp.sum(cp.abs(combined) ** 2))
                return gain
            else:
                combined = v.conj() @ self._Phi + self._h_d.conj()[cp.newaxis, :]
                gains = cp.sum(cp.abs(combined) ** 2, axis=1)
                return cp.asnumpy(gains)
        except Exception:
            # Fall back to CPU on any GPU error
            return self._batch_cpu(theta_batch, use_practical)

    def _batch_cpu(self, theta_batch, use_practical):
        """CPU fallback using NumPy (mirrors objective.compute_channel_gain)."""
        from phase_shift_model import reflection_vector

        if theta_batch.ndim == 1:
            v = reflection_vector(theta_batch, use_practical)
            combined = v.conj() @ self._Phi + self._h_d.conj()
            return np.sum(np.abs(combined) ** 2)
        else:
            v = reflection_vector(theta_batch, use_practical)
            combined = v.conj() @ self._Phi + self._h_d.conj()[np.newaxis, :]
            return np.sum(np.abs(combined) ** 2, axis=1)
