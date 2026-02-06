"""Normality test tool."""

import pandas as pd
from scipy import stats


def check_normality(df: pd.DataFrame, column: str) -> str:
    """
    Check if a column follows a normal distribution using the Shapiro-Wilk test.
    
    Args:
        df: The dataframe
        column: The numeric column to test
        
    Returns:
        Normality test results with interpretation
    """
    if column not in df.columns:
        return f"Error: Column '{column}' not found."
    
    data = df[column].dropna()
    
    if len(data) < 3:
        return "Need at least 3 non-null values for normality test."
    
    # Shapiro-Wilk (limit to 5000 samples)
    sample = data.sample(n=min(5000, len(data)), random_state=42)
    stat, p_value = stats.shapiro(sample)
    
    # Skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    is_normal = p_value >= 0.05
    
    return f"""Normality Test for '{column}'

Shapiro-Wilk statistic: {stat:.4f}
p-value: {p_value:.4f}

Skewness: {skewness:.3f} (0 = symmetric)
Kurtosis: {kurtosis:.3f} (0 = normal tails)

Interpretation: The data {'appears to be' if is_normal else 'does NOT appear to be'} normally distributed (α=0.05).
{'This supports using parametric tests.' if is_normal else 'Consider non-parametric tests or data transformation.'}"""
