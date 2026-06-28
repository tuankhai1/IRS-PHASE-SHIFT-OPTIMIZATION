"""
Configuration for the IRS Phase Shift Optimization project.

The system parameters follow:
S. Abeywickrama et al., "Intelligent Reflecting Surface: Practical Phase Shift
Model and Beamforming Optimization".
"""

import numpy as np

# ============================================================
# System Parameters
# ============================================================
M = 2                  # Number of AP antennas
N_DEFAULT = 40         # Default number of IRS reflecting elements

PT_DBM = 36            # Transmit power at AP (dBm)
SIGMA2_DBM = -94       # Noise power (dBm)

# Convert to linear scale (Watts)
PT = 10 ** (PT_DBM / 10) * 1e-3
SIGMA2 = 10 ** (SIGMA2_DBM / 10) * 1e-3

# ============================================================
# Geometry
# ============================================================
D_AP_IRS = 500         # AP-to-IRS horizontal distance (m)
D_VERTICAL = 2         # Vertical distance between AP-user and AP-IRS lines (m)

# ============================================================
# Path Loss Model
# ============================================================
REF_LOSS_DB = 40
C0 = 10 ** (-REF_LOSS_DB / 10)   # Reference path loss at 1 m

ALPHA_AI = 2.2   # AP -> IRS
ALPHA_IU = 2.8   # IRS -> user
ALPHA_AU = 3.8   # AP -> user

# ============================================================
# Practical Phase Shift Model
# beta(theta) = (1 - beta_min) * ((sin(theta - phi) + 1) / 2)^k + beta_min
# ============================================================
BETA_MIN = 0.2
K_PARAM = 1.6
PHI_PARAM = 0.43 * np.pi

# ============================================================
# Simulation Settings
# ============================================================
NUM_REALIZATIONS = 1000
SEED = 42

# ============================================================
# Algorithm Parameters
# ============================================================

# --- Alternating Optimization (AO) ---
AO_MAX_ITER = 100           # Maximum AO outer iterations
AO_TOL = 1e-6               # Convergence tolerance (relative change)
AO_1D_SEARCH_POINTS = 1000  # Number of points for 1D exhaustive search

# --- Particle Swarm Optimization (PSO) ---
# Standard global-best PSO.
PSO_POP_SIZE = 50           # Number of particles
PSO_MAX_ITER = 200          # Maximum iterations
PSO_INERTIA = 0.729         # Inertia weight
PSO_C1 = 1.49445            # Cognitive coefficient
PSO_C2 = 1.49445            # Social coefficient
PSO_V_MAX = np.pi           # Velocity clamp (radians)

# --- CMA-ES ---
CMAES_MAX_ITER = 300        # Maximum generations
CMAES_SIGMA0 = np.pi        # Initial step size covers [-pi, pi]
CMAES_TOL = 1e-8            # Convergence tolerance
