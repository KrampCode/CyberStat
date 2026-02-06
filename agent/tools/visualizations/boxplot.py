"""Box plot visualization tool with statistics."""

import io
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import THEME, CYBER_PALETTE, format_label


def create_box_plot(
    df: pd.DataFrame, 
    column: str, 
    group_column: Optional[str] = None
) -> tuple[bytes, str, dict]:
    """
    Create a box plot for a numeric column, optionally grouped.
    Returns statistics that the agent can use for accurate analysis.
    
    Args:
        df: The dataframe
        column: The numeric column to plot
        group_column: Optional categorical column to group by
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    stats = {}
    
    # Boxplot styling
    boxprops = dict(facecolor=THEME['neon_blue'], color=THEME['neon_blue'], alpha=0.7)
    medianprops = dict(color=THEME['neon_pink'], linewidth=2)
    whiskerprops = dict(color=THEME['text_secondary'])
    capprops = dict(color=THEME['text_secondary'])
    flierprops = dict(marker='o', markerfacecolor=THEME['neon_orange'], markersize=5, alpha=0.6)
    
    if group_column and group_column in df.columns:
        groups = df[group_column].unique()
        data_to_plot = [df[df[group_column] == g][column].dropna().values for g in groups]
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=groups,
                       boxprops=boxprops, medianprops=medianprops,
                       whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
        
        # Color each box differently
        for patch, color in zip(bp['boxes'], CYBER_PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{format_label(column)} by {format_label(group_column)}')
        ax.set_xlabel(format_label(group_column), color='#ffffff', fontweight='medium')
        desc = f"Box plot of {format_label(column)} grouped by {format_label(group_column)}"
        
        # Calculate per-group statistics
        grouped = df.groupby(group_column)[column]
        stats["grouped_by"] = group_column
        stats["groups"] = {}
        
        for group_name, group_data in grouped:
            group_data = group_data.dropna()
            if len(group_data) > 0:
                q1, median, q3 = np.percentile(group_data, [25, 50, 75])
                stats["groups"][str(group_name)] = {
                    "count": int(len(group_data)),
                    "mean": round(float(group_data.mean()), 2),
                    "median": round(float(median), 2),
                    "std": round(float(group_data.std()), 2),
                    "min": round(float(group_data.min()), 2),
                    "max": round(float(group_data.max()), 2),
                    "q1": round(float(q1), 2),
                    "q3": round(float(q3), 2),
                    "iqr": round(float(q3 - q1), 2)
                }
    else:
        bp = ax.boxplot(df[column].dropna().values, patch_artist=True,
                       boxprops=boxprops, medianprops=medianprops,
                       whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
        ax.set_title(format_label(column))
        ax.set_xticklabels([format_label(column)])
        desc = f"Box plot showing distribution of {format_label(column)}"
        
        # Calculate overall statistics
        data = df[column].dropna()
        if len(data) > 0:
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            stats["overall"] = {
                "count": int(len(data)),
                "mean": round(float(data.mean()), 2),
                "median": round(float(median), 2),
                "std": round(float(data.std()), 2),
                "min": round(float(data.min()), 2),
                "max": round(float(data.max()), 2),
                "q1": round(float(q1), 2),
                "q3": round(float(q3), 2),
                "iqr": round(float(q3 - q1), 2)
            }
    
    ax.set_ylabel(format_label(column), color='#ffffff', fontweight='medium')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue(), desc, stats
