"""Scatter plot visualization tool with statistics."""

import io
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from . import THEME, format_label


def create_scatter_plot(
    df: pd.DataFrame, 
    x_column: str, 
    y_column: str,
    color_column: Optional[str] = None
) -> tuple[bytes, str, dict]:
    """
    Create a scatter plot of two numeric columns.
    Returns correlation and regression statistics for accurate analysis.
    
    Args:
        df: The dataframe
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column to color points by
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found")
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found")
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    if color_column and color_column in df.columns:
        # Create a cyberpunk colormap
        from matplotlib.colors import LinearSegmentedColormap
        cyber_cmap = LinearSegmentedColormap.from_list('cyber', 
            [THEME['neon_blue'], THEME['neon_purple'], THEME['neon_pink']])
        
        scatter = ax.scatter(df[x_column], df[y_column], 
                           c=df[color_column].astype('category').cat.codes, 
                           alpha=0.7, cmap=cyber_cmap, s=50, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax, label=color_column)
        cbar.ax.yaxis.label.set_color(THEME['text'])
        desc = f"Scatter plot of {y_column} vs {x_column}, colored by {color_column}"
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.7, color=THEME['neon_blue'], 
                  s=50, edgecolors='none')
        desc = f"Scatter plot of {y_column} vs {x_column}"
    
    # Get clean data for statistics
    clean_df = df[[x_column, y_column]].dropna()
    x_data = clean_df[x_column]
    y_data = clean_df[y_column]
    
    stats = {
        "x_column": x_column,
        "y_column": y_column,
        "n_points": int(len(clean_df)),
    }
    
    # Add trend line and calculate statistics
    if len(clean_df) > 2:
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_data, y_data)
        
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=THEME['neon_pink'], linestyle='--', linewidth=2, 
               alpha=0.9, label=f'r = {r_value:.2f}')
        ax.legend(loc='best', framealpha=0.9)
        
        # Interpret correlation strength
        r_abs = abs(r_value)
        if r_abs < 0.3:
            strength = "weak"
        elif r_abs < 0.5:
            strength = "moderate"
        elif r_abs < 0.7:
            strength = "moderately strong"
        else:
            strength = "strong"
        
        direction = "positive" if r_value > 0 else "negative"
        
        stats["correlation"] = {
            "r": round(float(r_value), 3),
            "r_squared": round(float(r_value ** 2), 3),
            "p_value": round(float(p_value), 4),
            "significant": p_value < 0.05,
            "strength": strength,
            "direction": direction,
            "interpretation": f"{strength} {direction} correlation"
        }
        
        stats["regression"] = {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "std_error": round(float(std_err), 4),
            "interpretation": f"For every 1-unit increase in {x_column}, {y_column} changes by {slope:.2f}"
        }
    
    ax.set_xlabel(format_label(x_column), color='#ffffff', fontweight='medium')
    ax.set_ylabel(format_label(y_column), color='#ffffff', fontweight='medium')
    ax.set_title(f'{format_label(y_column)} vs {format_label(x_column)}')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue(), desc, stats
