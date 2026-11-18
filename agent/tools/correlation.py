"""Correlation calculation tool."""

import pandas as pd

from .derived import select_numeric


def calculate_correlation(df: pd.DataFrame, method: str = "pearson") -> str:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: The dataframe
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
        
    Returns:
        Correlation matrix as formatted string, plus notable correlations
    """
    numeric_df = select_numeric(df)
    
    if numeric_df.empty:
        return "No numeric columns found for correlation analysis."
    
    if len(numeric_df.columns) < 2:
        return "Need at least 2 numeric columns for correlation analysis."
    
    corr_matrix = numeric_df.corr(method=method).round(3)
    
    # Find notable correlations (strong positive or negative)
    notable = []
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i+1:]:
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) >= 0.7:
                strength = "strong" if abs(corr_val) >= 0.8 else "moderate-strong"
                direction = "positive" if corr_val > 0 else "negative"
                notable.append(f"  {col1} & {col2}: {corr_val:.3f} ({strength} {direction})")
    
    result = f"Correlation Matrix ({method}):\n{corr_matrix.to_string()}"
    
    if notable:
        result += "\n\nNotable correlations (|r| >= 0.7):\n" + "\n".join(notable)
    else:
        result += "\n\nNo strong correlations (|r| >= 0.7) found."
    
    return result
