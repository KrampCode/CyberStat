"""T-test tool."""

import pandas as pd
import numpy as np
from scipy import stats


def perform_ttest(
    df: pd.DataFrame, 
    column: str, 
    group_column: str
) -> str:
    """
    Perform an independent samples t-test comparing a numeric column across two groups.
    
    Args:
        df: The dataframe
        column: The numeric column to compare
        group_column: The categorical column defining the two groups
        
    Returns:
        T-test results with interpretation
    """
    if column not in df.columns:
        return f"Error: Column '{column}' not found."
    if group_column not in df.columns:
        return f"Error: Group column '{group_column}' not found."
    
    groups = df[group_column].dropna().unique()
    if len(groups) != 2:
        return f"T-test requires exactly 2 groups. Found {len(groups)}: {list(groups)}"
    
    group1_data = df[df[group_column] == groups[0]][column].dropna()
    group2_data = df[df[group_column] == groups[1]][column].dropna()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((group1_data.std()**2 + group2_data.std()**2) / 2)
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Interpret
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    effect_interp = "negligible" if abs(cohens_d) < 0.2 else \
                    "small" if abs(cohens_d) < 0.5 else \
                    "medium" if abs(cohens_d) < 0.8 else "large"
    
    return f"""T-Test Results: {column} by {group_column}

Group '{groups[0]}': n={len(group1_data)}, mean={group1_data.mean():.2f}, std={group1_data.std():.2f}
Group '{groups[1]}': n={len(group2_data)}, mean={group2_data.mean():.2f}, std={group2_data.std():.2f}

t-statistic: {t_stat:.3f}
p-value: {p_value:.4f}
Cohen's d: {cohens_d:.3f} ({effect_interp} effect)

Interpretation: The difference is {significance} with a {effect_interp} effect size."""
