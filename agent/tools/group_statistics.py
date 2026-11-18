"""Group statistics tool for comparing numeric values across categories."""

import pandas as pd
import numpy as np


def get_group_statistics(
    df: pd.DataFrame, 
    numeric_column: str, 
    group_column: str
) -> str:
    """
    Get detailed statistics for a numeric column grouped by a categorical column.
    Essential for making accurate comparisons between groups.
    
    Args:
        df: The dataframe
        numeric_column: The numeric column to analyze
        group_column: The categorical column to group by
        
    Returns:
        Detailed per-group statistics as formatted string
    """
    if numeric_column not in df.columns:
        return f"Error: Column '{numeric_column}' not found."
    if group_column not in df.columns:
        return f"Error: Column '{group_column}' not found."
    
    # Check if numeric column is actually numeric
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        return f"Error: Column '{numeric_column}' is not numeric."
    
    grouped = df.groupby(group_column)[numeric_column]
    
    result_lines = [
        f"Group Statistics: {numeric_column} by {group_column}",
        "=" * 60,
        ""
    ]
    
    # Calculate stats for each group
    group_stats = []
    for group_name, group_data in grouped:
        data = group_data.dropna()
        if len(data) == 0:
            continue
            
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        
        stats = {
            "name": group_name,
            "count": len(data),
            "mean": data.mean(),
            "median": median,
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "q1": q1,
            "q3": q3
        }
        group_stats.append(stats)
    
    # Sort groups by mean for easy comparison
    group_stats.sort(key=lambda x: x["mean"], reverse=True)
    
    for stats in group_stats:
        result_lines.extend([
            f"  {stats['name']}:",
            f"   Count:  {stats['count']:,}",
            f"   Mean:   {stats['mean']:.2f}",
            f"   Median: {stats['median']:.2f}",
            f"   Std:    {stats['std']:.2f}",
            f"   Range:  {stats['min']:.2f} - {stats['max']:.2f}",
            f"   IQR:    {stats['q1']:.2f} - {stats['q3']:.2f}",
            ""
        ])
    
    # Add comparison summary
    if len(group_stats) >= 2:
        result_lines.append("=" * 60)
        result_lines.append("COMPARISON SUMMARY:")
        result_lines.append("")
        
        means = [(s["name"], s["mean"]) for s in group_stats]
        highest = means[0]
        lowest = means[-1]
        
        if highest[1] != lowest[1]:
            diff = highest[1] - lowest[1]
            pct_diff = (diff / abs(lowest[1]) * 100) if lowest[1] != 0 else float('inf')
            
            result_lines.append(f"Highest mean: {highest[0]} ({highest[1]:.2f})")
            result_lines.append(f"Lowest mean:  {lowest[0]} ({lowest[1]:.2f})")
            result_lines.append(f"Difference:   {diff:.2f} ({pct_diff:.1f}% higher)")
            result_lines.append("")
            
            # Rank all groups
            result_lines.append("Ranking by mean (highest to lowest):")
            for i, (name, mean) in enumerate(means, 1):
                result_lines.append(f"  {i}. {name}: {mean:.2f}")
    
    return "\n".join(result_lines)
