"""
Main entry point for the IRS Phase Shift Optimization Project.

Runs all simulation scenarios and generates figures.

Usage:
    python main.py                 # Full simulation (1000 realizations)
    python main.py --realizations 20
    python main.py --fig 5         # Run only Fig. 5
    python main.py --fig 6         # Run only Fig. 6
    python main.py --fig 7         # Run only Fig. 7
    python main.py --fig 12        # Run fixed-component ablation
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from simulation import (run_simulation_fig5, run_simulation_fig6,
                        run_simulation_fig7, run_simulation_fig8,
                        run_simulation_fig9, run_simulation_fig10,
                        run_simulation_fig11)
from fixed_component_simulation import run_simulation_fig12
from plot_results import (plot_fig5, plot_fig6, plot_fig7,
                          plot_fig8, plot_fig9, plot_fig10,
                          plot_fig11)
from plot_fixed_component import plot_fig12
from config import NUM_REALIZATIONS, SEED


def main():
    parser = argparse.ArgumentParser(
        description='IRS practical phase-shift simulation'
    )
    parser.add_argument('--realizations', type=int, default=NUM_REALIZATIONS,
                        help='Number of channel realizations')
    parser.add_argument('--fig', type=int, choices=[5, 6, 7, 8, 9, 10, 11, 12],
                        default=None,
                        help='Run only a specific figure (5-12)')
    args = parser.parse_args()

    # Create output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  IRS Phase Shift Optimization Project")
    print(f"  Mode: paper figures + component-level optimization")
    print(f"  Channel realizations: {args.realizations}")
    print(f"  Base seed: {SEED}")
    print(f"  Output directory: {out_dir}")
    print(f"{'#'*60}")

    figs_to_run = [args.fig] if args.fig else [5, 6, 7, 8, 9, 10, 11, 12]

    # ---- Fig. 5 ----  
    if 5 in figs_to_run:
        results5 = run_simulation_fig5(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig5.npz')
        )
        plot_fig5(results5,
                  save_path=os.path.join(out_dir, 'fig5_rate_vs_distance.png'))
        print_summary(results5, 'Fig. 5', 'd_values')
        print_runtime_summary(results5, 'Fig. 5')
        save_runtime_table(results5, 'd_values',
                           save_path=os.path.join(out_dir, 'runtime_table_fig5.md'))

    # ---- Fig. 6 ----
    if 6 in figs_to_run:
        results6 = run_simulation_fig6(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig6.npz')
        )
        plot_fig6(results6,
                  save_path=os.path.join(out_dir, 'fig6_rate_vs_N.png'))
        print_summary(results6, 'Fig. 6', 'N_values')
        print_runtime_summary(results6, 'Fig. 6')
        save_runtime_table(results6, 'N_values',
                           save_path=os.path.join(out_dir, 'runtime_table_fig6.md'))

    # ---- Fig. 7 ----
    if 7 in figs_to_run:
        results7 = run_simulation_fig7(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig7.npz')
        )
        plot_fig7(results7,
                  save_path=os.path.join(out_dir, 'fig7_discrete_phases.png'))
        print_summary(results7, 'Fig. 7', 'd_values')
        print_runtime_summary(results7, 'Fig. 7')
        save_runtime_table(results7, 'd_values',
                           save_path=os.path.join(out_dir, 'runtime_table_fig7.md'))

    # ---- Fig. 8 ----
    if 8 in figs_to_run:
        results8 = run_simulation_fig8(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig8.npz')
        )
        plot_fig8(results8,
                  save_path=os.path.join(out_dir, 'fig8_component_vs_distance.png'))
        print_summary(results8, 'Fig. 8', 'd_values')
        print_runtime_summary(results8, 'Fig. 8')
        save_runtime_table(results8, 'd_values',
                           save_path=os.path.join(out_dir, 'runtime_table_fig8.md'))

    # ---- Fig. 9 ----
    if 9 in figs_to_run:
        results9 = run_simulation_fig9(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig9.npz')
        )
        plot_fig9(results9,
                  save_path=os.path.join(out_dir, 'fig9_component_vs_N.png'))
        print_summary(results9, 'Fig. 9', 'N_values')
        print_runtime_summary(results9, 'Fig. 9')
        save_runtime_table(results9, 'N_values',
                           save_path=os.path.join(out_dir, 'runtime_table_fig9.md'))

    # ---- Fig. 10 ----
    if 10 in figs_to_run:
        results10 = run_simulation_fig10(
            num_realizations=min(args.realizations, 20),
            save_path=os.path.join(out_dir, 'results_fig10.npz')
        )
        plot_fig10(results10,
                   save_path=os.path.join(out_dir, 'fig10_convergence.png'))

    # ---- Fig. 11 ----
    if 11 in figs_to_run:
        results11 = run_simulation_fig11(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig11.npz')
        )
        plot_fig11(results11,
                   save_path=os.path.join(out_dir, 'fig11_phase_vs_component.png'))
        print_summary(results11, 'Fig. 11', 'd_values')
        print_runtime_summary(results11, 'Fig. 11')
        save_runtime_table(results11, 'd_values',
                           save_path=os.path.join(out_dir, 'runtime_table_fig11.md'))

    # ---- Fig. 12 ----
    if 12 in figs_to_run:
        results12 = run_simulation_fig12(
            num_realizations=args.realizations,
            save_path=os.path.join(out_dir, 'results_fig12.npz')
        )
        plot_fig12(results12,
                   save_path=os.path.join(out_dir, 'fig12_fixed_component_ablation.png'))
        print_summary(results12, 'Fig. 12', 'd_values')

    print(f"\n{'#'*60}")
    print(f"  All simulations complete!")
    print(f"  Figures saved in: {out_dir}")
    print(f"{'#'*60}\n")


def print_summary(results, fig_name, x_key):
    """Print a summary table of results."""
    print(f"\n  --- {fig_name} Summary ---")
    x_vals = results[x_key]
    metadata_keys = {'seed'}
    scheme_keys = [
        k for k in results
        if k != x_key and k not in metadata_keys and not k.startswith('runtime_')
        and isinstance(results[k], np.ndarray)
        and results[k].shape == x_vals.shape
    ]

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


def print_runtime_summary(results, fig_name):
    """Print per-scheme runtime measured over the same realization seeds."""
    required = {
        'runtime_scheme_names',
        'runtime_overall_mean_seconds',
        'runtime_overall_total_seconds',
        'runtime_num_samples',
        'runtime_wall_seconds',
    }
    if not required.issubset(results):
        return

    schemes = results['runtime_scheme_names']
    mean_seconds = results['runtime_overall_mean_seconds']
    total_seconds = results['runtime_overall_total_seconds']
    num_samples = int(results['runtime_num_samples'])
    wall_seconds = float(results['runtime_wall_seconds'])

    print(f"  --- {fig_name} Runtime Summary ---")
    print(f"  Samples per scheme: {num_samples}")
    print(f"  Parallel wall time: {wall_seconds:.2f}s")
    print(f"  {'Scheme':<35} {'Mean/seed (s)':>14} {'Total CPU (s)':>14}")
    print("  " + "-" * 65)
    for scheme, mean_s, total_s in zip(schemes, mean_seconds, total_seconds):
        print(f"  {scheme:<35} {mean_s:>14.4f} {total_s:>14.2f}")
    print()


def save_runtime_table(results, x_key, save_path):
    """Save a Markdown runtime table with one row per scheme."""
    required = {
        'runtime_scheme_names',
        'runtime_mean_seconds',
        'runtime_overall_mean_seconds',
        'runtime_overall_total_seconds',
        'runtime_num_samples',
        'runtime_wall_seconds',
    }
    if not required.issubset(results):
        return

    x_vals = results[x_key]
    schemes = results['runtime_scheme_names']
    mean_by_x = results['runtime_mean_seconds']
    overall_mean = results['runtime_overall_mean_seconds']
    overall_total = results['runtime_overall_total_seconds']
    num_samples = int(results['runtime_num_samples'])
    num_realizations = num_samples // len(x_vals)
    wall_seconds = float(results['runtime_wall_seconds'])

    headers = ['Scheme']
    headers.extend(str(int(x)) if float(x).is_integer() else f'{x:g}' for x in x_vals)
    headers.extend(['Overall mean/seed (s)', 'Total CPU (s)'])

    lines = [
        f'Runtime table for `{x_key}`.',
        '',
        f'- Channel realizations per x-value: `{num_realizations}`',
        f'- Parallel wall time: `{wall_seconds:.2f} s`',
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join([':---'] + ['---:'] * (len(headers) - 1)) + ' |',
    ]

    for i, scheme in enumerate(schemes):
        row = [str(scheme)]
        row.extend(f'{value:.6f}' for value in mean_by_x[i])
        row.append(f'{overall_mean[i]:.6f}')
        row.append(f'{overall_total[i]:.2f}')
        lines.append('| ' + ' | '.join(row) + ' |')

    with open(save_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Runtime table saved to {save_path}")


if __name__ == '__main__':
    main()
