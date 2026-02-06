"""Histogram visualization tool with statistics."""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from . import THEME, format_label


def create_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> tuple[bytes, str, dict]:
    """
    Create a histogram for a numeric column.
    Returns statistics that the agent can use for accurate analysis.
    
    Args:
        df: The dataframe
        column: The numeric column to plot
        bins: Number of bins for the histogram
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    data = df[column].dropna()
    
    # Cyberpunk styled histogram
    ax.hist(data, bins=bins, edgecolor=THEME['plot_bg'], alpha=0.85, color=THEME['neon_blue'])
    ax.axvline(data.mean(), color=THEME['neon_pink'], linestyle='--', linewidth=2, 
               label=f'Mean: {data.mean():.2f}')
    ax.axvline(data.median(), color=THEME['neon_green'], linestyle='--', linewidth=2,
               label=f'Median: {data.median():.2f}')
    
    ax.set_xlabel(format_label(column), color='#ffffff', fontweight='medium')
    ax.set_ylabel('Frequency', color='#ffffff', fontweight='medium')
    ax.set_title(format_label(column))
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    # Calculate comprehensive statistics
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    
    # Skewness interpretation
    skewness = float(data.skew())
    if abs(skewness) < 0.5:
        skew_interpretation = "approximately symmetric"
    elif skewness > 0:
        skew_interpretation = "right-skewed (tail extends to higher values)"
    else:
        skew_interpretation = "left-skewed (tail extends to lower values)"
    
    stats = {
        "column": column,
        "count": int(len(data)),
        "mean": round(float(data.mean()), 2),
        "median": round(float(median), 2),
        "std": round(float(data.std()), 2),
        "min": round(float(data.min()), 2),
        "max": round(float(data.max()), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(q3 - q1), 2),
        "skewness": round(skewness, 2),
        "skew_interpretation": skew_interpretation,
        "range": round(float(data.max() - data.min()), 2)
    }
    
    return buf.getvalue(), f"Histogram showing distribution of {column}", stats
