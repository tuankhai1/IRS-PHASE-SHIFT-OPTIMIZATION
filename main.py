"""
Main entry point for the IRS Phase Shift Optimization Project.

Runs all simulation scenarios and generates figures.

Usage:
    python main.py                 # Full simulation (1000 realizations)
    python main.py --quick         # Quick test (20 realizations)
    python main.py --fig 5         # Run only Fig. 5
    python main.py --fig 6         # Run only Fig. 6
    python main.py --fig 7         # Run only Fig. 7
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

from simulation import run_simulation_fig5, run_simulation_fig6, run_simulation_fig7
from plot_results import plot_fig5, plot_fig6, plot_fig7
from config import NUM_REALIZATIONS


def main():
    parser = argparse.ArgumentParser(
        description='IRS Phase Shift Optimization: PSO & CMA-ES'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer realizations (20)')
    parser.add_argument('--realizations', type=int, default=None,
                        help='Number of channel realizations')
    parser.add_argument('--fig', type=int, choices=[5, 6, 7], default=None,
                        help='Run only a specific figure (5, 6, or 7)')
    args = parser.parse_args()

    # Determine number of realizations
    if args.realizations is not None:
        num_real = args.realizations
    elif args.quick:
        num_real = 20
    else:
        num_real = NUM_REALIZATIONS

    # Create output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  IRS Phase Shift Optimization Project")
    print(f"  Algorithms: AO (baseline) | PSO | CMA-ES")
    print(f"  Channel realizations: {num_real}")
    print(f"  Output directory: {out_dir}")
    print(f"{'#'*60}")

    figs_to_run = [args.fig] if args.fig else [5, 6, 7]

    # ---- Fig. 5 ----  
    if 5 in figs_to_run:
        results5 = run_simulation_fig5(
            num_realizations=num_real,
            save_path=os.path.join(out_dir, 'results_fig5.npz')
        )
        fig5 = plot_fig5(results5,
                         save_path=os.path.join(out_dir, 'fig5_rate_vs_distance.png'))
        print_summary(results5, 'Fig. 5', 'd_values')

    # ---- Fig. 6 ----
    if 6 in figs_to_run:
        results6 = run_simulation_fig6(
            num_realizations=num_real,
            save_path=os.path.join(out_dir, 'results_fig6.npz')
        )
        fig6 = plot_fig6(results6,
                         save_path=os.path.join(out_dir, 'fig6_rate_vs_N.png'))
        print_summary(results6, 'Fig. 6', 'N_values')

    # ---- Fig. 7 ----
    if 7 in figs_to_run:
        results7 = run_simulation_fig7(
            num_realizations=num_real,
            save_path=os.path.join(out_dir, 'results_fig7.npz')
        )
        fig7 = plot_fig7(results7,
                         save_path=os.path.join(out_dir, 'fig7_discrete_phases.png'))
        print_summary(results7, 'Fig. 7', 'd_values')

    print(f"\n{'#'*60}")
    print(f"  All simulations complete!")
    print(f"  Figures saved in: {out_dir}")
    print(f"{'#'*60}\n")


def print_summary(results, fig_name, x_key):
    """Print a summary table of results."""
    print(f"\n  --- {fig_name} Summary ---")
    x_vals = results[x_key]
    scheme_keys = [k for k in results if k != x_key and isinstance(results[k], np.ndarray)]

    # Header
    header = f"  {'Scheme':<35}"
    for x in x_vals:
        header += f" {x:>7.0f}"
    print(header)
    print("  " + "-" * (35 + 8 * len(x_vals)))

    # Rows
    for scheme in scheme_keys:
        row = f"  {scheme:<35}"
        for val in results[scheme]:
            row += f" {val:>7.2f}"
        print(row)
    print()


if __name__ == '__main__':
    main()