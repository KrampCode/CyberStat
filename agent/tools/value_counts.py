"""Value counts tool."""

import pandas as pd


def get_value_counts(df: pd.DataFrame, column: str, top_n: int = 20) -> str:
    """
    Get frequency counts for a column's values.
    
    Args:
        df: The dataframe
        column: The column to analyze
        top_n: Number of top values to show (default 20)
        
    Returns:
        Value counts with percentages
    """
    if column not in df.columns:
        return f"Error: Column '{column}' not found."
    
    counts = df[column].value_counts().head(top_n)
    percentages = (counts / len(df) * 100).round(1)
    
    result = f"Value Counts for '{column}' (top {min(top_n, len(counts))}):\n\n"
    for val, count in counts.items():
        pct = percentages[val]
        result += f"  {val}: {count} ({pct}%)\n"
    
    total_unique = df[column].nunique()
    if total_unique > top_n:
        result += f"\n  ... and {total_unique - top_n} more unique values"
    
    return result
