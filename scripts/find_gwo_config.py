"""
Test multiple GWO pop_size configs to find convergence sweet spot.

Runs a few channel realizations with different GWO population sizes
and plots convergence curves side-by-side with PSO baseline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from config import N_DEFAULT, SEED, COMP_PSO_POP_SIZE, COMP_PSO_MAX_ITER
from channel_model import generate_channels
from algorithms.pso_component import pso_component_optimize
from algorithms.gwo import gwo_component_optimize
from config import OMEGA as CIRCUIT_OMEGA, Z0


def test_config(N, d_horizontal, num_realizations, gwo_pop_sizes, max_iter, seed):
    """Test GWO with different pop_sizes and compare with PSO."""
    master_rng = np.random.default_rng(seed)

    # Storage for convergence histories
    pso_histories = []
    gwo_histories = {ps: [] for ps in gwo_pop_sizes}

    for r in range(num_realizations):
        task_seed = int(master_rng.integers(0, 2**31))
        chan_rng = np.random.default_rng(task_seed)
        h_d, Phi = generate_channels(N, d_horizontal, chan_rng)

        # PSO baseline (pop=100, fixed)
        pso_rng = np.random.default_rng(task_seed + 1)
        _, _, hist = pso_component_optimize(
            Phi, h_d, N, pop_size=COMP_PSO_POP_SIZE,
            max_iter=max_iter, rng=pso_rng, return_history=True)
        pso_histories.append(hist)

        # GWO with different pop_sizes
        for ps in gwo_pop_sizes:
            gwo_rng = np.random.default_rng(task_seed + 2)
            t0 = time.time()
            _, _, hist = gwo_component_optimize(
                Phi, h_d, N, pop_size=ps,
                max_iter=max_iter, rng=gwo_rng, return_history=True)
            elapsed = time.time() - t0
            gwo_histories[ps].append(hist)
            print(f"  Realization {r+1}/{num_realizations}, "
                  f"GWO pop={ps}: final={hist[-1]:.4f}, time={elapsed:.1f}s")

    # Average over realizations
    pso_avg = np.mean(pso_histories, axis=0)
    gwo_avgs = {ps: np.mean(gwo_histories[ps], axis=0) for ps in gwo_pop_sizes}

    return pso_avg, gwo_avgs


def plot_results(pso_avg, gwo_avgs, max_iter, save_path):
    """Plot convergence comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    iters = np.arange(len(pso_avg))

    ax.plot(iters, pso_avg, 'purple', linewidth=2,
            label=f'PSO (pop={COMP_PSO_POP_SIZE})')

    colors = ['cyan', 'lime', 'orange', 'red', 'blue']
    for i, (ps, avg) in enumerate(sorted(gwo_avgs.items())):
        ax.plot(iters, avg, color=colors[i % len(colors)],
                linewidth=2, linestyle='--',
                label=f'GWO (pop={ps}), final={avg[-1]:.3f}')

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Achievable rate $R_{SE}$ (bits/s/Hz)', fontsize=13)
    ax.set_title(f'GWO Pop Size Sweep (max_iter={max_iter})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    N = N_DEFAULT  # 40
    d_horizontal = 498
    num_realizations = 10  # fewer for speed
    max_iter = 2000

    # Test these GWO pop sizes
    gwo_pop_sizes = [500, 1000, 2000, 3000]

    print(f"Testing GWO convergence with pop_sizes={gwo_pop_sizes}")
    print(f"PSO baseline: pop={COMP_PSO_POP_SIZE}, iter={max_iter}")
    print(f"Realizations: {num_realizations}, N={N}, d={d_horizontal}m\n")

    t0 = time.time()
    pso_avg, gwo_avgs = test_config(
        N, d_horizontal, num_realizations, gwo_pop_sizes, max_iter, SEED)
    total = time.time() - t0

    print(f"\nTotal time: {total:.1f}s")
    print(f"\nFinal values at iteration {max_iter}:")
    print(f"  PSO (pop={COMP_PSO_POP_SIZE}): {pso_avg[-1]:.4f}")
    for ps in sorted(gwo_avgs):
        avg = gwo_avgs[ps]
        # Check if "converged" - last 10% change < 1%
        tail = avg[int(0.9 * len(avg)):]
        rel_change = (tail[-1] - tail[0]) / tail[-1] * 100
        status = "CONVERGED" if rel_change < 1.0 else f"still climbing ({rel_change:.1f}%)"
        beats_pso = "YES" if avg[-1] > pso_avg[-1] else "NO"
        print(f"  GWO (pop={ps}): {avg[-1]:.4f} | beats PSO: {beats_pso} | {status}")

    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'results', 'gwo_pop_sweep.png')
    plot_results(pso_avg, gwo_avgs, max_iter, save_path)
