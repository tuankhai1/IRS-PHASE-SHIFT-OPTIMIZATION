"""
Plotting utilities for generating publication-quality figures.

Reproduces the paper's Figs. 5, 6, and 7 with PSO and GWO curves.
Adds Figs. 8, 9, 10 for component-level optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


# ============================================================
# Phase-Level Scheme Styles (Figs. 5, 6)
# ============================================================
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
    'pso':                         {'label': '6) PSO, practical',
                                    'color': '#8E24AA', 'marker': 'v', 'ls': '--', 'lw': 1.8},
    'gwo':                         {'label': '7) GWO, practical',
                                    'color': '#00ACC1', 'marker': 'H', 'ls': '--', 'lw': 1.8},
}

# ============================================================
# Component-Level Scheme Styles (Figs. 8, 9)
# ============================================================
COMPONENT_STYLES = {
    'pso_component':     {'label': 'PSO component-level',
                          'color': '#8E24AA', 'marker': 'v', 'ls': '-', 'lw': 2.2},

    'gwo_component':     {'label': 'GWO component-level',
                          'color': '#00ACC1', 'marker': 'H', 'ls': '-', 'lw': 2.2},
    'hybrid_component':  {'label': 'Hybrid (AO+PSO) component',
                          'color': '#FB8C00', 'marker': '*', 'ls': '-', 'lw': 2.8},
    'hybrid_pso_pso_component': {'label': 'Hybrid PSO-PSO component',
                                 'color': '#1E88E5', 'marker': 'P', 'ls': '--', 'lw': 2.4},
    'hybrid_pso_gwo_component': {'label': 'Hybrid PSO-GWO component',
                                 'color': '#43A047', 'marker': 'X', 'ls': '-.', 'lw': 2.4},
}

# ============================================================
# Fig. 7: Discrete Phase Shift Styles
# ============================================================
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

# ============================================================
# Convergence Styles (Fig. 10)
# ============================================================
CONVERGENCE_STYLES = {
    'pso_component':  {'label': 'PSO component-level',
                       'color': '#8E24AA', 'ls': '-', 'lw': 2.2},

    'gwo_component':  {'label': 'GWO component-level',
                       'color': '#00ACC1', 'ls': '-.', 'lw': 2.2},
}


def _setup_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.minorticks_off()
        
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    
    ax.grid(False, which='minor')
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6, color='#A0A0A0')
    
    ax.tick_params(direction='in', top=True, right=True, labelsize=11, pad=6)


def plot_fig5(results, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    d_values = results['d_values']

    plot_order = [
        'upper_bound', 'ao_practical_prop1', 'ao_practical_1d',
        'ideal_design_practical_eval',
        'pso', 'gwo',
        'lower_bound'
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
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig6(results, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    N_values = results['N_values']

    plot_order = [
        'upper_bound', 'ao_practical_prop1', 'ao_practical_1d',
        'ideal_design_practical_eval',
        'pso', 'gwo',
        'lower_bound'
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
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig7(results, save_path=None):
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


# ============================================================
# Component-Level Figures
# ============================================================

def plot_fig8(results, save_path=None):
    """Fig. 8: Component-level rate vs. AP-user distance."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    d_values = results['d_values']

    plot_order = [
        'upper_bound', 'ao_practical_prop1',
        'pso_component', 'gwo_component',
        'lower_bound',
    ]

    styles = {**SCHEME_STYLES, **COMPONENT_STYLES}

    for scheme in plot_order:
        if scheme in results and scheme in styles:
            style = styles[scheme]
            ax.plot(d_values, results[scheme],
                    label=style['label'], color=style['color'],
                    marker=style.get('marker', 'o'),
                    linestyle=style['ls'],
                    linewidth=style['lw'], markersize=8, clip_on=False)

    ax.set_xlim([d_values[0], d_values[-1]])
    _setup_axes(ax,
                xlabel='AP-user horizontal distance: $d$ (m)',
                ylabel='Achievable rate (bits/s/Hz)',
                title='Fig. 8: Component-Level Optimization vs. Phase-Level (N=40)')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig9(results, save_path=None):
    """Fig. 9: Component-level rate vs. number of IRS elements."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    N_values = results['N_values']

    plot_order = [
        'upper_bound', 'ao_practical_prop1',
        'pso_component', 'gwo_component',
        'lower_bound',
    ]

    styles = {**SCHEME_STYLES, **COMPONENT_STYLES}

    for scheme in plot_order:
        if scheme in results and scheme in styles:
            style = styles[scheme]
            ax.plot(N_values, results[scheme],
                    label=style['label'], color=style['color'],
                    marker=style.get('marker', 'o'),
                    linestyle=style['ls'],
                    linewidth=style['lw'], markersize=8, clip_on=False)

    ax.set_xlim([N_values[0], N_values[-1]])
    _setup_axes(ax,
                xlabel='Number of reflecting elements ($N$)',
                ylabel='Achievable rate (bits/s/Hz)',
                title='Fig. 9: Component-Level Optimization vs. Phase-Level (d=498m)')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig10(results, save_path=None):
    """Fig. 10: Convergence comparison of component-level algorithms."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    iterations = results['iterations']

    for scheme in ['pso_component', 'gwo_component']:
        if scheme in results and scheme in CONVERGENCE_STYLES:
            style = CONVERGENCE_STYLES[scheme]
            ax.plot(iterations, results[scheme],
                    label=style['label'], color=style['color'],
                    linestyle=style['ls'], linewidth=style['lw'])

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Achievable rate $R_{SE}$ (bits/s/Hz)', fontsize=13)
    ax.set_title('Fig. 10: Convergence of Component-Level Algorithms\n'
                 '($N$=40, $d$=498m)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, linestyle='-', linewidth=0.6, alpha=0.6, color='#A0A0A0')
    ax.tick_params(direction='in', top=True, right=True, labelsize=11, pad=6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig


def plot_fig11(results, save_path=None):
    """Fig. 11: Focused comparison — phase vs. component vs. hybrid.

    Plots component-level and hybrid schemes over the Fig. 5 distance range.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    d_values = results['d_values']

    # Combined styles
    styles = {**SCHEME_STYLES, **COMPONENT_STYLES}

    # Build focused plot order
    plot_order = [
        'upper_bound',
        'pso_component',
        'gwo_component',
        'hybrid_component',
        'hybrid_pso_pso_component',
        'hybrid_pso_gwo_component',
        'lower_bound',
    ]

    for scheme in plot_order:
        if scheme in results and scheme in styles:
            style = styles[scheme]
            ms = 11 if scheme.startswith('hybrid') else 8
            ax.plot(d_values, results[scheme],
                    label=style['label'], color=style['color'],
                    marker=style.get('marker', 'o'),
                    linestyle=style['ls'],
                    linewidth=style['lw'], markersize=ms, clip_on=False)

    ax.set_xlim([d_values[0], d_values[-1]])
    _setup_axes(ax,
                xlabel='AP-user horizontal distance: $d$ (m)',
                ylabel='Achievable rate (bits/s/Hz)',
                title='Fig. 11: Phase-Level vs. Component-Level vs. Hybrid (N=40)')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig
