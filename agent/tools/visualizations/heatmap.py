"""Correlation heatmap visualization tool with statistics."""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from . import THEME, format_label


def create_correlation_heatmap(df: pd.DataFrame) -> tuple[bytes, str, dict]:
    """
    Create a correlation heatmap for all numeric columns.
    Returns top correlations that the agent can use for accurate analysis.
    
    Args:
        df: The dataframe
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation heatmap")
    
    corr_matrix = numeric_df.corr()
    
    # Rename columns and index to formatted labels
    formatted_names = {col: format_label(col) for col in corr_matrix.columns}
    corr_matrix = corr_matrix.rename(columns=formatted_names, index=formatted_names)
    
    # Adjust figure size based on number of columns (compact for grid display)
    n_cols = len(corr_matrix.columns)
    fig_size = max(5, min(8, n_cols * 0.6))  # Cap size for grid layout
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    
    # Create cyberpunk diverging colormap: pink (negative) -> dark -> blue (positive)
    cyber_cmap = LinearSegmentedColormap.from_list('cyber_diverge', 
        [THEME['neon_pink'], THEME['plot_bg'], THEME['neon_blue']])
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap=cyber_cmap, 
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 8, 'color': THEME['text']},
        linewidths=1,
        linecolor=THEME['grid']
    )
    
    ax.set_title('Correlations')
    
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=THEME['text'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=THEME['text'])
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    # Build statistics with all correlations sorted by strength
    correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i+1:]:
            r = corr_matrix.loc[col1, col2]
            correlations.append({
                "var1": col1,
                "var2": col2,
                "r": round(float(r), 3),
                "abs_r": abs(float(r))
            })
    
    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: x["abs_r"], reverse=True)
    
    # Classify correlations
    strong = [c for c in correlations if c["abs_r"] >= 0.7]
    moderate = [c for c in correlations if 0.4 <= c["abs_r"] < 0.7]
    weak = [c for c in correlations if c["abs_r"] < 0.4]
    
    stats = {
        "n_variables": len(corr_matrix.columns),
        "variables": list(corr_matrix.columns),
        "all_correlations": correlations[:20],  # Top 20 by strength
        "strong_correlations": [
            {"pair": f"{c['var1']} & {c['var2']}", "r": c["r"]} for c in strong
        ],
        "moderate_correlations": [
            {"pair": f"{c['var1']} & {c['var2']}", "r": c["r"]} for c in moderate
        ],
        "summary": {
            "strong_count": len(strong),
            "moderate_count": len(moderate),
            "weak_count": len(weak)
        }
    }
    
    return buf.getvalue(), "Correlation heatmap showing relationships between numeric variables", stats
