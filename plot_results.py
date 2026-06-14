"""
Plotting utilities for generating publication-quality figures.

Reproduces the paper's Figs. 5, 6, and 7 with additional PSO and CMA-ES curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


# ============================================================
# Color scheme — uses highly distinct, paper-style colors
# ============================================================

# Main figure styles (Figs. 5, 6)
SCHEME_STYLES = {
    'upper_bound':                 {'label': '1) Upper bound (ideal IRS)',
                                    'color': '#E53935', 'marker': 's', 'ls': '-.', 'lw': 2.5},
    'ao_practical_prop1':          {'label': '2) AO, practical (Prop. 1)',
                                    'color': '#43A047', 'marker': 'o', 'ls': '-', 'lw': 2.2},
    'ao_practical_1d':             {'label': '3) AO, practical (1D search)',
                                    'color': '#1E88E5', 'marker': '^', 'ls': '--', 'lw': 2.2},
    'ideal_design_practical_eval': {'label': '4) Ideal design, practical eval',
                                    'color': '#FB8C00', 'marker': 'D', 'ls': ':', 'lw': 2.5},
    'lower_bound':                 {'label': '5) Lower bound (no IRS)',
                                    'color': '#212121', 'marker': '*', 'ls': '--', 'lw': 2.0},
    'pso_practical':               {'label': '6) PSO, practical',
                                    'color': '#8E24AA', 'marker': 'v', 'ls': '-', 'lw': 2.0},
    'cmaes_practical':             {'label': '7) CMA-ES, practical',
                                    'color': '#00ACC1', 'marker': 'P', 'ls': '-', 'lw': 2.0},
}

# Fig 7: vibrant Material Design 500 colors
# Practical (solid lines): warm saturated tones
# Ideal (dashed lines): cool saturated tones with distinct markers
DISCRETE_STYLES = {
    'ao_practical_discrete_1': {'label': 'Practical, b=1',
                                'color': '#F44336', 'marker': 'o', 'ls': '-', 'lw': 2.5, 'ms': 9},
    'ao_practical_discrete_2': {'label': 'Practical, b=2',
                                'color': '#FF9800', 'marker': 's', 'ls': '-', 'lw': 2.5, 'ms': 9},
    'ao_practical_discrete_3': {'label': 'Practical, b=3',
                                'color': '#4CAF50', 'marker': '^', 'ls': '-', 'lw': 2.5, 'ms': 9},
    'ao_ideal_discrete_1':     {'label': 'Ideal, b=1',
                                'color': '#2196F3', 'marker': 'x', 'ls': '--', 'lw': 2.2, 'ms': 10},
    'ao_ideal_discrete_2':     {'label': 'Ideal, b=2',
                                'color': '#9C27B0', 'marker': '+', 'ls': '--', 'lw': 2.2, 'ms': 10},
    'ao_ideal_discrete_3':     {'label': 'Ideal, b=3',
                                'color': '#00BCD4', 'marker': 'D', 'ls': '--', 'lw': 2.2, 'ms': 8},
}


def _setup_axes(ax, xlabel, ylabel, title=None):
    """Apply consistent styling to axes."""
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
        
    # Ensure minor ticks are completely disabled
    ax.minorticks_off()
        
    # Y-axis intervals of 0.5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    
    # Align the first and last grid lines exactly with the bounding box spines
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    
    # Neater grids: solid light major grid only, explicitly turn off minor grid
    ax.grid(False, which='minor')
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6, color='#A0A0A0')
    
    # MATLAB style: ticks point inward and appear on all 4 sides, full bounding box
    ax.tick_params(direction='in', top=True, right=True, labelsize=11, pad=6)


def plot_fig5(results, save_path=None):
    """
    Plot Fig. 5: Achievable rate vs. AP-user horizontal distance.

    Parameters
    ----------
    results : dict
        Output from run_simulation_fig5().
    save_path : str, optional
        If provided, saves the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    d_values = results['d_values']

    plot_order = [
        'upper_bound', 'ao_practical_prop1', 'ao_practical_1d',
        'ideal_design_practical_eval',
        'pso_practical', 'cmaes_practical', 'lower_bound'
    ]

    for scheme in plot_order:
        if scheme in results and scheme in SCHEME_STYLES:
            style = SCHEME_STYLES[scheme]
            ax.plot(d_values, results[scheme],
                    label=style['label'], color=style['color'],
                    marker=style['marker'], linestyle=style['ls'],
                    linewidth=style['lw'], markersize=8, clip_on=False)

    ax.set_xlim([d_values[0], d_values[-1]])
    _setup_axes(ax,
                xlabel='AP-user horizontal distance: $d$ (m)',
                ylabel='Achievable rate (bits/s/Hz)',
                title=f'Fig. 5: Achievable Rate vs. Distance (N={results.get("N", 40)})')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig6(results, save_path=None):
    """
    Plot Fig. 6: Achievable rate vs. number of reflecting elements.

    Parameters
    ----------
    results : dict
        Output from run_simulation_fig6().
    save_path : str, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    N_values = results['N_values']

    plot_order = [
        'upper_bound', 'ao_practical_prop1', 'ao_practical_1d',
        'ideal_design_practical_eval',
        'pso_practical', 'cmaes_practical', 'lower_bound'
    ]

    for scheme in plot_order:
        if scheme in results and scheme in SCHEME_STYLES:
            style = SCHEME_STYLES[scheme]
            ax.plot(N_values, results[scheme],
                    label=style['label'], color=style['color'],
                    marker=style['marker'], linestyle=style['ls'],
                    linewidth=style['lw'], markersize=8, clip_on=False)

    ax.set_xlim([N_values[0], N_values[-1]])
    _setup_axes(ax,
                xlabel='Number of reflecting elements ($N$)',
                ylabel='Achievable rate (bits/s/Hz)',
                title='Fig. 6: Achievable Rate vs. $N$ (d=498m)')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig7(results, save_path=None):
    """
    Plot Fig. 7: Achievable rate vs. distance with discrete phase shifts.

    Parameters
    ----------
    results : dict
        Output from run_simulation_fig7().
    save_path : str, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    d_values = results['d_values']

    # Plot upper and lower bounds
    for scheme in ['upper_bound', 'lower_bound']:
        if scheme in results:
            style = SCHEME_STYLES[scheme]
            ax.plot(d_values, results[scheme],
                    label=style['label'], color=style['color'],
                    marker=style['marker'], linestyle=style['ls'],
                    linewidth=style['lw'], markersize=8, clip_on=False)

    # Plot discrete schemes — practical first (solid), then ideal (dashed)
    plot_order = [
        'ao_practical_discrete_1', 'ao_practical_discrete_2', 'ao_practical_discrete_3',
        'ao_ideal_discrete_1', 'ao_ideal_discrete_2', 'ao_ideal_discrete_3',
    ]

    for key in plot_order:
        if key in results and key in DISCRETE_STYLES:
            style = DISCRETE_STYLES[key]
            ax.plot(d_values, results[key],
                    label=style['label'], color=style['color'],
                    marker=style['marker'], linestyle=style['ls'],
                    linewidth=style['lw'], markersize=style['ms'], clip_on=False)

    ax.set_xlim([d_values[0], d_values[-1]])
    _setup_axes(ax,
                xlabel='AP-user horizontal distance: $d$ (m)',
                ylabel='Achievable rate (bits/s/Hz)',
                title='Fig. 7: Achievable Rate with Discrete Phase Shifts ($N$=40)')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig
