"""Test of independence between two categorical columns (chi-square / Fisher's exact)."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import association

from .multiple_testing import ALPHA, InferenceResult, PValue

MAX_LEVELS = 20
# Chi-square is unreliable when more than 20% of cells have expected counts below 5 (Cochran)
MIN_EXPECTED = 5
MAX_SMALL_EXPECTED_SHARE = 0.2
# Adjusted residuals beyond +/-2 mark cells that deviate notably from independence
RESIDUAL_THRESHOLD = 2.0


def interpret_cramers_v(v: float, min_dim: int) -> str:
    """Cohen's thresholds for Cramer's V, which depend on min(rows, cols) - 1."""
    df_star = min(min_dim - 1, 3)
    small, medium, large = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29)}[df_star]
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def adjusted_residuals(observed: pd.DataFrame, expected: np.ndarray) -> pd.DataFrame:
    """Haberman's adjusted standardized residuals: approximately N(0, 1) under independence."""
    n = observed.to_numpy().sum()
    row_share = observed.sum(axis=1).to_numpy()[:, None] / n
    col_share = observed.sum(axis=0).to_numpy()[None, :] / n
    residuals = (observed.to_numpy() - expected) / np.sqrt(expected * (1 - row_share) * (1 - col_share))
    return pd.DataFrame(residuals, index=observed.index, columns=observed.columns)


def independence_test(df: pd.DataFrame, column_a: str, column_b: str) -> InferenceResult:
    """
    Test whether two categorical columns are associated.

    Uses the chi-square test of independence, or Fisher's exact test for 2x2 tables
    with small expected counts. Reports Cramer's V as the effect size and the cells
    that drive any association.

    Args:
        df: The dataframe
        column_a: First categorical column
        column_b: Second categorical column

    Returns:
        Test results with effect size and interpretation, plus the p-value for
        session-wide multiple-testing correction
    """
    for col in (column_a, column_b):
        if col not in df.columns:
            return InferenceResult(f"Error: Column '{col}' not found.")
    if column_a == column_b:
        return InferenceResult("Error: Choose two different columns.")

    observed = pd.crosstab(df[column_a], df[column_b])
    n_rows, n_cols = observed.shape
    if n_rows < 2 or n_cols < 2:
        return InferenceResult("Error: Both columns need at least 2 distinct values.")
    if n_rows > MAX_LEVELS or n_cols > MAX_LEVELS:
        return InferenceResult(
            f"Error: Table is {n_rows}x{n_cols}; at most {MAX_LEVELS} levels per column are supported. "
            "Is one of these columns continuous? Bin it or use compare_groups instead."
        )

    n = int(observed.to_numpy().sum())
    chi2, chi2_p, dof, expected = stats.chi2_contingency(observed)
    small_share = float((expected < MIN_EXPECTED).mean())
    cramers_v = float(association(observed, method="cramer"))
    effect = interpret_cramers_v(cramers_v, min(n_rows, n_cols))

    lines = [
        f"TEST OF INDEPENDENCE: {column_a} x {column_b}",
        f"Contingency table ({n_rows}x{n_cols}, n={n:,}):",
        observed.to_string(),
        "",
    ]

    use_fisher = (n_rows, n_cols) == (2, 2) and (expected < MIN_EXPECTED).any()
    if use_fisher:
        odds_ratio, p_value = stats.fisher_exact(observed)
        test_name = "Fisher's exact test"
        lines += [
            "TEST SELECTED: Fisher's exact test (2x2 table with expected counts below 5)",
            f"Odds ratio: {odds_ratio:.3f}",
            f"p-value: {p_value:.4g}",
        ]
    else:
        p_value = chi2_p
        test_name = "Chi-square test of independence"
        correction = " with Yates' continuity correction" if dof == 1 else ""
        lines += [
            f"TEST SELECTED: Chi-square test of independence{correction}",
            f"Chi-square({dof}) = {chi2:.3f}, p = {p_value:.4g}",
        ]
        if small_share > MAX_SMALL_EXPECTED_SHARE:
            lines.append(
                f"Warning: {small_share:.0%} of cells have expected counts below {MIN_EXPECTED} "
                "(more than 20%), so the chi-square approximation is unreliable. "
                "Consider combining rare categories."
            )

    lines.append(f"Cramer's V: {cramers_v:.3f} ({effect} association)")

    significant = p_value < ALPHA
    lines.append(
        f"\nInterpretation: {column_a} and {column_b} are "
        f"{'significantly associated' if significant else 'not significantly associated'} at alpha={ALPHA}, "
        f"with a {effect} effect size."
    )

    if significant:
        residuals = adjusted_residuals(observed, expected)
        notable = residuals.stack()
        notable = notable[notable.abs() > RESIDUAL_THRESHOLD].sort_values(key=abs, ascending=False)
        if not notable.empty:
            lines.append("\nCELLS DRIVING THE ASSOCIATION (adjusted residual beyond +/-2):")
            for (a_val, b_val), r in notable.head(6).items():
                direction = "more" if r > 0 else "fewer"
                lines.append(
                    f"  {column_a}={a_val}, {column_b}={b_val}: observed {observed.loc[a_val, b_val]}, "
                    f"expected {expected[observed.index.get_loc(a_val), observed.columns.get_loc(b_val)]:.1f} "
                    f"({direction} than expected, residual {r:+.2f})"
                )

    hypothesis = f"{column_a} is associated with {column_b}"
    return InferenceResult("\n".join(lines), [PValue(test_name, hypothesis, float(p_value))])
