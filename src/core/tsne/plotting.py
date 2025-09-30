"""t-SNE plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_plotting_style(figure_config: dict) -> None:
    """Setup matplotlib plotting style from config."""
    plt.rcParams.update({
        'font.size': figure_config['font_size'],
        'axes.labelsize': figure_config['axes_labelsize'],
        'axes.titlesize': figure_config['axes_titlesize'],
        'xtick.labelsize': figure_config['xtick_labelsize'],
        'ytick.labelsize': figure_config['ytick_labelsize'],
        'legend.fontsize': figure_config['legend_fontsize'],
        'figure.titlesize': figure_config['figure_titlesize'],
        'figure.dpi': figure_config['dpi']
    })


def plot_qc_comparison(
    embeddings: dict, metadata: dict, plots_dir: Path, complexity: int, qc_threshold: int
) -> None:
    """Plot before/after QC comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get consistent color scale
    vmin = min(metadata['preqc']['surface_holes'].min(), metadata['postqc']['surface_holes'].min())
    vmax = metadata['preqc']['surface_holes'].max()

    # Pre-QC plot
    scatter1 = axes[0].scatter(
        embeddings['preqc'][:, 0], embeddings['preqc'][:, 1],
        c=metadata['preqc']['surface_holes'], cmap='gray_r',
        alpha=0.7, s=8, vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f'Before QC (n = {len(embeddings["preqc"]):,})', fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Post-QC plot
    scatter2 = axes[1].scatter(
        embeddings['postqc'][:, 0], embeddings['postqc'][:, 1],
        c=metadata['postqc']['surface_holes'], cmap='gray_r',
        alpha=0.7, s=8, vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f'After QC (n = {len(embeddings["postqc"]):,})', fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Colorbar
    plt.tight_layout()
    cbar = fig.colorbar(scatter1, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Surface Topology Defects', rotation=270, labelpad=20)

    plt.suptitle('Quality Control Impact on Data Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(plots_dir / f'qc_comparison_complexity{complexity}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print stats
    preqc_count = len(embeddings['preqc'])
    postqc_count = len(embeddings['postqc'])
    removed = preqc_count - postqc_count
    print(f"QC removed {removed:,} samples ({removed/preqc_count*100:.1f}%) with >{qc_threshold} surface defects")


def plot_comparison(
    embeddings: dict, metadata: dict, data_type1: str, data_type2: str,
    grouping: str, title: str, colors: dict, filename: str,
    plots_dir: Path, complexity: int, plot_config: dict
) -> None:
    """Plot comparison between two data types."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (dtype, ax) in enumerate(zip([data_type1, data_type2], axes)):
        groups = metadata[dtype][grouping]
        embedding = embeddings[dtype]

        for group in np.unique(groups):
            if pd.isna(group):
                continue
            mask = groups == group
            color = colors.get(group, '#666666')

            if grouping == 'research_groups':
                alpha = plot_config['alpha']['control'] if group == 'Control' else plot_config['alpha']['highlighted']
                size = plot_config['size']['control'] if group == 'Control' else plot_config['size']['highlighted']
                edge = plot_config['edge']['control'] if group == 'Control' else plot_config['edge']['highlighted']
                linewidth = plot_config['linewidth']['control'] if group == 'Control' else plot_config['linewidth']['highlighted']
            else:
                alpha = 0.7
                size = 8
                edge = None
                linewidth = 0

            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, alpha=alpha, s=size, label=group,
                edgecolors=edge, linewidths=linewidth
            )

        ax.set_title(f'{dtype.title()} (n = {len(embedding):,})', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        if i == 0:
            ax.set_ylabel('t-SNE 2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 1:  # Only add legend to second plot
            # Create custom legend with uniform marker sizes
            handles = []
            for group in np.unique(groups):
                if not pd.isna(group):
                    color = colors.get(group, '#666666')
                    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=10,
                                            label=group, markeredgecolor='black',
                                            markeredgewidth=0.5))

            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=12, markerscale=1.2, frameon=False)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / f'{filename}_complexity{complexity}.png', dpi=300, bbox_inches='tight')
    plt.show()