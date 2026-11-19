"""T-test tool."""

import pandas as pd
import numpy as np
from scipy import stats

from .multiple_testing import InferenceResult, PValue


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d using the n-weighted pooled standard deviation."""
    n1, n2 = len(a), len(b)
    pooled_var = ((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def hedges_g(d: float, n1: int, n2: int) -> float:
    """Small-sample bias correction of Cohen's d."""
    return d * (1 - 3 / (4 * (n1 + n2) - 9))


def interpret_effect_size(d: float) -> str:
    """Cohen's conventional thresholds for |d|."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def perform_ttest(
    df: pd.DataFrame,
    column: str,
    group_column: str,
    equal_var: bool = False,
    alpha: float = 0.05
) -> InferenceResult:
    """
    Perform an independent samples t-test comparing a numeric column across two groups.

    Defaults to Welch's t-test, which does not assume equal variances and loses
    little power when variances are in fact equal.

    Args:
        df: The dataframe
        column: The numeric column to compare
        group_column: The categorical column defining the two groups
        equal_var: Use Student's pooled-variance t-test instead of Welch's
        alpha: Significance level; the confidence interval is at 1 - alpha

    Returns:
        T-test results with confidence interval, effect size, and interpretation,
        plus the p-value for session-wide multiple-testing correction
    """
    if column not in df.columns:
        return InferenceResult(f"Error: Column '{column}' not found.")
    if group_column not in df.columns:
        return InferenceResult(f"Error: Group column '{group_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        return InferenceResult(f"Error: Column '{column}' is not numeric.")

    # Sort so the direction of the difference is deterministic
    groups = sorted(df[group_column].dropna().unique(), key=str)
    if len(groups) != 2:
        return InferenceResult(f"T-test requires exactly 2 groups. Found {len(groups)}: {list(groups)}")

    group1_data = df[df[group_column] == groups[0]][column].dropna()
    group2_data = df[df[group_column] == groups[1]][column].dropna()
    n1, n2 = len(group1_data), len(group2_data)
    if n1 < 2 or n2 < 2:
        return InferenceResult(f"T-test requires at least 2 observations per group. Found n={n1} and n={n2}.")

    result = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
    ci = result.confidence_interval(confidence_level=1 - alpha)
    mean_diff = group1_data.mean() - group2_data.mean()

    d = cohens_d(group1_data, group2_data)
    g = hedges_g(d, n1, n2)
    effect_interp = interpret_effect_size(d)

    test_name = "Student's t-test (equal variances)" if equal_var else "Welch's t-test (unequal variances)"
    significant = result.pvalue < alpha
    ci_pct = f"{(1 - alpha) * 100:g}%"

    text = f"""T-Test Results: {column} by {group_column}
Test: {test_name}

Group '{groups[0]}': n={n1}, mean={group1_data.mean():.2f}, std={group1_data.std():.2f}
Group '{groups[1]}': n={n2}, mean={group2_data.mean():.2f}, std={group2_data.std():.2f}

Mean difference ('{groups[0]}' - '{groups[1]}'): {mean_diff:.2f}
{ci_pct} CI for the difference: [{ci.low:.2f}, {ci.high:.2f}]
t-statistic: {result.statistic:.3f} (df = {result.df:.1f})
p-value: {result.pvalue:.4g}
Cohen's d: {d:.3f} ({effect_interp} effect) | Hedges' g: {g:.3f}

Interpretation: The difference is {'statistically significant' if significant else 'not statistically significant'} at alpha={alpha} with a {effect_interp} effect size.
{'The CI excludes 0.' if significant else 'The CI includes 0, so no difference is a plausible explanation of the data.'}
Report the CI and effect size alongside the p-value - statistical significance alone does not imply practical importance."""
    
    hypothesis = f"{column} differs between '{groups[0]}' and '{groups[1]}' ({group_column})"
    test_short = "Student's t-test" if equal_var else "Welch's t-test"
    return InferenceResult(text, [PValue(test_short, hypothesis, float(result.pvalue))])
