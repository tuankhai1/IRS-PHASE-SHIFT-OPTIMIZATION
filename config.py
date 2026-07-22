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

# --- Adaptive PSO (APSO) for Phase-Shift ---
# Ref: Y. Shi, R. Eberhart, "A modified particle swarm optimizer,"
#      Proc. IEEE ICEC, pp. 69-73, 1998.
# NOTE: Uses Shi-Eberhart coefficients (c1=c2=2.0), NOT Clerc's
#       constriction coefficients (1.49445). The adaptive inertia
#       w: 0.9→0.4 is designed for c1=c2=2.0.
APSO_POP_SIZE = 50             # Number of particles
APSO_MAX_ITER = 200            # Maximum iterations
APSO_W_MAX = 0.9               # Initial inertia (exploration)
APSO_W_MIN = 0.4               # Final inertia (exploitation)
APSO_C1 = 2.0                  # Cognitive coefficient (Shi-Eberhart)
APSO_C2 = 2.0                  # Social coefficient (Shi-Eberhart)

# --- Grey Wolf Optimizer (GWO) ---
# Ref: S. Mirjalili et al., "Grey Wolf Optimizer,"
#      Advances in Engineering Software, vol. 69, pp. 46-61, 2014.
GWO_POP_SIZE = 50              # Number of wolves
GWO_MAX_ITER = 200             # Maximum iterations

# ============================================================
# Circuit Model Parameters (Component-Level Optimization)
# ============================================================
# Operating frequency: 5.8 GHz (ISM band)
# Ref: ITU Radio Regulations Art. 5, Footnote 5.150;
#      C. Liaskos et al., "A New Wireless Communication Paradigm
#      through Software-Controlled Metasurfaces," IEEE Commun. Mag., 2018.
FREQ = 5.8e9                   # Operating frequency (Hz)
OMEGA = 2 * np.pi * FREQ      # Angular frequency (rad/s)
Z0 = 377.0                    # Free-space impedance (Ω)

# Component bounds based on typical varactor-based IRS elements.
# Ref: Skyworks SMV1231-079LF datasheet; extended for design coverage.
L1_BOUNDS = (0.5e-9, 5.0e-9)  # Coupling inductance (H)
L2_BOUNDS = (0.1e-9, 3.0e-9)  # Varactor series inductance (H)
C_BOUNDS  = (0.1e-12, 5.0e-12) # Varactor capacitance (F)
R_BOUNDS  = (0.5, 5.0)        # Varactor series resistance (Ω)

# --- Component-level PSO ---
COMP_PSO_POP_SIZE = 100        # Larger population for 4N-dim space
COMP_PSO_MAX_ITER = 800        # More iterations for convergence

# --- Component-level Adaptive PSO (APSO) ---
# Ref: Y. Shi, R. Eberhart, "A modified particle swarm optimizer,"
#      Proc. IEEE ICEC, pp. 69-73, 1998.
COMP_APSO_POP_SIZE = 100
COMP_APSO_MAX_ITER = 800       # More iterations for convergence
COMP_APSO_W_MAX = 0.9         # Initial inertia (exploration)
COMP_APSO_W_MIN = 0.4         # Final inertia (exploitation)

# --- Component-level GWO ---
COMP_GWO_POP_SIZE = 100
COMP_GWO_MAX_ITER = 800        # More iterations for convergence
