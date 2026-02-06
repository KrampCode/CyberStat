"""PCA visualization tools."""

import io
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from . import THEME, CYBER_PALETTE, format_label


def create_pca_scatter(
    df: pd.DataFrame,
    pc1_col: str = 'PC1',
    pc2_col: str = 'PC2',
    color_column: Optional[str] = None,
    title: str = "PCA Projection"
) -> tuple[bytes, str, dict]:
    """
    Create a scatter plot of the first two principal components.
    
    Args:
        df: Dataframe with PC columns
        pc1_col: Name of first PC column
        pc2_col: Name of second PC column
        color_column: Optional column to color points by
        title: Plot title
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    if pc1_col not in df.columns or pc2_col not in df.columns:
        raise ValueError(f"PC columns not found. Available: {list(df.columns)}")
    
    # Get clean data
    plot_df = df[[pc1_col, pc2_col]].dropna()
    if color_column and color_column in df.columns:
        plot_df = df[[pc1_col, pc2_col, color_column]].dropna()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if color_column and color_column in df.columns:
        # Color by category or cluster
        if df[color_column].dtype in ['object', 'category'] or df[color_column].nunique() <= 10:
            categories = sorted(plot_df[color_column].unique())
            
            for i, cat in enumerate(categories):
                mask = plot_df[color_column] == cat
                ax.scatter(
                    plot_df.loc[mask, pc1_col], 
                    plot_df.loc[mask, pc2_col],
                    c=[CYBER_PALETTE[i % len(CYBER_PALETTE)]], 
                    label=str(cat),
                    alpha=0.7,
                    s=50,
                    edgecolors='none'
                )
            ax.legend(title=format_label(color_column), bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        else:
            cyber_cmap = LinearSegmentedColormap.from_list('cyber', 
                [THEME['neon_blue'], THEME['neon_purple'], THEME['neon_pink']])
            scatter = ax.scatter(
                plot_df[pc1_col], 
                plot_df[pc2_col],
                c=plot_df[color_column],
                alpha=0.7,
                cmap=cyber_cmap,
                s=50,
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax, label=format_label(color_column))
            cbar.ax.yaxis.label.set_color(THEME['text'])
        desc = f"PCA scatter plot colored by {color_column}"
    else:
        ax.scatter(plot_df[pc1_col], plot_df[pc2_col], alpha=0.7, color=THEME['neon_blue'], 
                  s=50, edgecolors='none')
        desc = "PCA scatter plot of first two components"
    
    ax.set_xlabel(format_label(pc1_col), color='#ffffff', fontweight='medium')
    ax.set_ylabel(format_label(pc2_col), color='#ffffff', fontweight='medium')
    ax.set_title(format_label(title))
    ax.axhline(y=0, color=THEME['text_secondary'], linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=0, color=THEME['text_secondary'], linestyle='--', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    # Statistics
    stats = {
        "n_points": len(plot_df),
        "pc1_range": [round(float(plot_df[pc1_col].min()), 2), round(float(plot_df[pc1_col].max()), 2)],
        "pc2_range": [round(float(plot_df[pc2_col].min()), 2), round(float(plot_df[pc2_col].max()), 2)],
    }
    
    if color_column and color_column in df.columns:
        stats["colored_by"] = color_column
        stats["n_categories"] = int(plot_df[color_column].nunique())
    
    return buf.getvalue(), desc, stats


def create_scree_plot(explained_variance: dict) -> tuple[bytes, str, dict]:
    """
    Create a scree plot showing explained variance by component.
    
    Args:
        explained_variance: Dict from PCA stats with variance info
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    components = list(explained_variance.keys())
    variances = [explained_variance[c]["percentage"] for c in components]
    cumulative = [explained_variance[c]["cumulative"] for c in components]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    x = range(len(components))
    
    # Bar chart for individual variance
    bars = ax1.bar(x, variances, color=THEME['neon_blue'], alpha=0.8, label='Individual',
                   edgecolor=THEME['plot_bg'])
    ax1.set_xlabel('Principal Component', color='#ffffff', fontweight='medium')
    ax1.set_ylabel('Explained Variance (%)', color=THEME['neon_blue'], fontweight='medium')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.tick_params(axis='y', labelcolor=THEME['neon_blue'])
    
    # Line for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color=THEME['neon_pink'], marker='o', linewidth=2, 
            markersize=8, label='Cumulative')
    ax2.set_ylabel('Cumulative Variance (%)', color=THEME['neon_pink'], fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=THEME['neon_pink'])
    ax2.axhline(y=90, color=THEME['neon_green'], linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_title('Variance Explained')
    
    # Add value labels on bars
    for bar, val in zip(bars, variances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color=THEME['text'])
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    stats = {
        "n_components": len(components),
        "total_variance_explained": cumulative[-1] if cumulative else 0,
        "variance_by_component": {c: v for c, v in zip(components, variances)}
    }
    
    return buf.getvalue(), "Scree plot showing variance explained by each principal component", stats
