"""Bar chart visualization tool with statistics."""

import io
import pandas as pd
import matplotlib.pyplot as plt
from . import THEME, CYBER_PALETTE, format_label


def create_bar_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> tuple[bytes, str, dict]:
    """
    Create a bar chart showing value counts for a column.
    Returns statistics that the agent can use for accurate analysis.
    
    Args:
        df: The dataframe
        column: The column to plot
        top_n: Number of top values to show
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    value_counts = df[column].value_counts().head(top_n)
    total = len(df[column].dropna())
    
    # Create gradient colors from the cyber palette
    colors = [CYBER_PALETTE[i % len(CYBER_PALETTE)] for i in range(len(value_counts))]
    
    bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors, alpha=0.85,
                   edgecolor=THEME['plot_bg'])
    ax.set_yticks(range(len(value_counts)))
    ax.set_yticklabels(value_counts.index)
    ax.invert_yaxis()  # Largest at top
    
    ax.set_xlabel('Count', color='#ffffff', fontweight='medium')
    ax.set_ylabel(format_label(column), color='#ffffff', fontweight='medium')
    ax.set_title(format_label(column))
    
    # Add value labels on bars
    for bar, val in zip(bars, value_counts.values):
        ax.text(val + (value_counts.values[0] * 0.02), bar.get_y() + bar.get_height()/2, 
               str(val), va='center', color=THEME['text'], fontsize=8)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    # Build statistics dictionary
    stats = {
        "column": column,
        "total_records": total,
        "unique_values": int(df[column].nunique()),
        "value_counts": {}
    }
    
    for val, count in value_counts.items():
        pct = round(count / total * 100, 1) if total > 0 else 0
        stats["value_counts"][str(val)] = {
            "count": int(count),
            "percentage": pct
        }
    
    return buf.getvalue(), f"Bar chart showing top {len(value_counts)} values in {column}", stats
