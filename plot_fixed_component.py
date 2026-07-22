"""
Plotting utilities for the fixed-component ablation study.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


FIXED_COMPONENT_STYLES = {
    'full': {
        'label': 'Full optimization (L1, L2, C, R)',
        'color': '#8E24AA',
        'marker': 'o',
        'ls': '-',
        'lw': 2.5,
    },
    'fix_R': {
        'label': 'Fix R; optimize L1, L2, C',
        'color': '#1E88E5',
        'marker': 's',
        'ls': '--',
        'lw': 2.2,
    },
    'fix_CR': {
        'label': 'Fix C, R; optimize L1, L2',
        'color': '#43A047',
        'marker': '^',
        'ls': '-.',
        'lw': 2.2,
    },
    'fix_L2CR': {
        'label': 'Fix L2, C, R; optimize L1',
        'color': '#FB8C00',
        'marker': 'D',
        'ls': ':',
        'lw': 2.5,
    },
    'lower_bound': {
        'label': 'Lower bound (no IRS)',
        'color': '#212121',
        'marker': '*',
        'ls': '--',
        'lw': 2.0,
    },
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
    ax.grid(True, which='major', linestyle='-', linewidth=0.6,
            alpha=0.6, color='#A0A0A0')
    ax.tick_params(direction='in', top=True, right=True, labelsize=11, pad=6)


def plot_fig12(results, save_path=None):
    """Fig. 12: Fixed-component ablation vs. full optimization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    d_values = results['d_values']

    plot_order = [
        'full',
        'fix_R',
        'fix_CR',
        'fix_L2CR',
        'lower_bound',
    ]

    for scheme in plot_order:
        if scheme in results and scheme in FIXED_COMPONENT_STYLES:
            style = FIXED_COMPONENT_STYLES[scheme]
            ax.plot(
                d_values,
                results[scheme],
                label=style['label'],
                color=style['color'],
                marker=style['marker'],
                linestyle=style['ls'],
                linewidth=style['lw'],
                markersize=8,
                clip_on=False,
            )

    ax.set_xlim([d_values[0], d_values[-1]])
    _setup_axes(
        ax,
        xlabel='AP-user horizontal distance: $d$ (m)',
        ylabel='Achievable rate (bits/s/Hz)',
        title='Fig. 12: Fixed-Component Ablation (N=40)',
    )
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")

    return fig
