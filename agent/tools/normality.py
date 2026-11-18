"""Normality test tool."""

import pandas as pd
from scipy import stats

SHAPIRO_MAX_N = 5000
LARGE_SAMPLE_N = 500
SMALL_SAMPLE_N = 20


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
    
    if not pd.api.types.is_numeric_dtype(data):
        return f"Error: Column '{column}' is not numeric."
    if len(data) < 3:
        return "Need at least 3 non-null values for normality test."
    
    # Shapiro-Wilk p-values are unreliable above 5000 samples, so test a random subsample
    sample = data.sample(n=min(SHAPIRO_MAX_N, len(data)), random_state=42)
    stat, p_value = stats.shapiro(sample)
    
    # Skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    is_normal = p_value >= 0.05
    
    notes = []
    if len(data) > SHAPIRO_MAX_N:
        notes.append(f"Test run on a random subsample of {SHAPIRO_MAX_N:,} of {len(data):,} values.")
    if len(sample) >= LARGE_SAMPLE_N:
        notes.append(
            f"With n={len(sample):,}, Shapiro-Wilk detects even trivial departures from normality. "
            "Judge practical normality by skewness and kurtosis (|skew| < 1 and |kurtosis| < 2 is usually "
            "fine for t-tests, which are robust to non-normality at this sample size by the central limit theorem)."
        )
    if len(sample) < SMALL_SAMPLE_N:
        notes.append(f"With only n={len(sample)}, the test has little power - a non-significant result is weak evidence of normality.")
    notes_text = ("\n\nNotes:\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""
    
    return f"""Normality Test for '{column}'

n = {len(data):,}
Shapiro-Wilk statistic: {stat:.4f}
p-value: {p_value:.4f}

Skewness: {skewness:.3f} (0 = symmetric)
Kurtosis: {kurtosis:.3f} (0 = normal tails)

Interpretation: The data {'appears to be' if is_normal else 'does NOT appear to be'} normally distributed (alpha=0.05).
{'This supports using parametric tests.' if is_normal else 'Consider non-parametric tests or data transformation.'}{notes_text}"""
