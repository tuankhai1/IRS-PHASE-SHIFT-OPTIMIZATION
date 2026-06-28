"""
Configuration file for IRS Phase Shift Optimization Project.
All parameters match the reference paper:
  S. Abeywickrama et al., "Intelligent Reflecting Surface: 
  Practical Phase Shift Model and Beamforming Optimization"
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
PT = 10 ** (PT_DBM / 10) * 1e-3          # ~3.981 W
SIGMA2 = 10 ** (SIGMA2_DBM / 10) * 1e-3  # ~3.981e-13 W

# ============================================================
# Geometry
# ============================================================
D_AP_IRS = 500         # AP-to-IRS horizontal distance (m)
D_VERTICAL = 2         # Vertical distance between AP-User line and AP-IRS line (m)

# ============================================================
# Path Loss Model
# ============================================================
# Signal attenuation at reference distance of 1m is 40 dB
REF_LOSS_DB = 40
C0 = 10 ** (-REF_LOSS_DB / 10)   # Reference path loss at 1m = 1e-4

# Path loss exponents
ALPHA_AI = 2.2   # AP  -> IRS
ALPHA_IU = 2.8   # IRS -> User
ALPHA_AU = 3.8   # AP  -> User

# ============================================================
# Practical Phase Shift Model:  β(θ) = (1-β_min)*((sin(θ-φ)+1)/2)^k + β_min
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
# Uses the Clerc-Kennedy constriction factor (requires c1+c2 > 4).
# Standard recommendation: c1 = c2 = 2.05, giving χ ≈ 0.7298.
PSO_POP_SIZE = 50           # Number of particles
PSO_MAX_ITER = 200          # Maximum iterations
PSO_C1 = 2.05               # Cognitive coefficient (constriction requirement)
PSO_C2 = 2.05               # Social coefficient  (constriction requirement)
PSO_V_MAX = np.pi           # Velocity clamp (radians)

# --- CMA-ES ---
CMAES_MAX_ITER = 300        # Maximum generations
CMAES_SIGMA0 = np.pi        # Initial step size (covers [-π, π])
CMAES_TOL = 1e-8            # Convergence tolerance
