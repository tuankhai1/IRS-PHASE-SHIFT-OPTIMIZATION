# IRS Phase Shift Optimization

## Introduction
This project simulates and optimizes the achievable spectrum efficiency (rate) of an Intelligent Reflecting Surface (IRS)-aided wireless communication system. It investigates the impact of realistic phase shift models—where the reflection amplitude is coupled with the phase shift—and compares various optimization algorithms. 

The framework evaluates system performance across multiple scenarios, taking into account continuous versus discrete phase shifts, varying distances between the Access Point (AP) and the user, and varying numbers of reflecting elements.

## Reference Paper
This project implements the system models, channel models, and optimization schemes from recent literature on practical IRS phase shift modeling. It demonstrates the differences between an "ideal" reflection model (constant amplitude) and a "practical" model (phase-dependent amplitude).

## The Approach
To maximize the achievable rate, the optimization of the IRS phase shifts is modeled as a non-convex optimization problem. We solve this using multiple algorithmic approaches:
1. **Alternating Optimization (AO)**: A highly optimized, coordinate-descent-based baseline leveraging Numba JIT compilation for maximum CPU performance.
2. **Particle Swarm Optimization (PSO)**: A population-based meta-heuristic algorithm using multi-strategy initialization, ring topologies, and constriction factors for robust multi-modal search.
3. **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**: An advanced evolutionary strategy that adaptively updates the search distribution.

*Note: Both PSO and CMA-ES support optional CuPy-based GPU acceleration via `gpu_backend.py` for high-throughput population evaluation.*

## Roadmap (Repository Structure)

- `main.py`: The main entry point. Runs all simulation scenarios and generates figures.
- `simulation.py`: Handles multiprocessing and parallelizing independent channel realizations across CPU cores.
- `objective.py`: Defines the objective functions, including effective channel gain and achievable rate.
- `channel_model.py`: Generates the direct (AP-user) and reflected (AP-IRS-user) fading channels.
- `phase_shift_model.py`: Mathematical models for both ideal and practical phase shifts.
- `numba_kernels.py`: JIT-compiled kernels to accelerate the inner loops of the AO algorithm.
- `gpu_backend.py`: CuPy-based GPU acceleration backend for batch processing meta-heuristic algorithms.
- `algorithms/`:
  - `ao.py`: Alternating Optimization algorithm.
  - `pso.py`: Particle Swarm Optimization algorithm.
  - `cmaes.py`: CMA-ES algorithm.

## How to Apply (Usage)

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install numpy matplotlib numba scipy
```
*(Optional: Install `cupy` if you want to use the GPU backend for PSO/CMA-ES).*

### Running Simulations
You can run the full suite of simulations (1000 channel realizations) by simply executing:
```bash
python main.py
```

For a quick test to verify everything is working (runs only 20 realizations):
```bash
python main.py --quick
```

To run a specific figure from the simulation results:
```bash
python main.py --fig 5  # Fig. 5: Rate vs. AP-user horizontal distance
python main.py --fig 6  # Fig. 6: Rate vs. number of reflecting elements
python main.py --fig 7  # Fig. 7: Rate vs. distance with discrete phase shifts
```

Outputs will be saved as numpy arrays (`.npz`) and plotted figures (`.png`) in the generated `results/` directory.
