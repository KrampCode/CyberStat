"""
Assumption-aware group comparison.

Checks each group's distribution, then picks the test:

                    approximately normal      not normal
    2 groups        Welch's t-test            Mann-Whitney U
    3+ groups       Welch's ANOVA             Kruskal-Wallis

Welch variants are used throughout because they do not assume equal variances and
lose little power when variances happen to be equal. When a k-group omnibus test is
significant, pairwise post-hoc tests are run with a Holm correction.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from .multiple_testing import ALPHA, InferenceResult, PValue, holm
from .ttest import perform_ttest

METHODS = ("auto", "welch", "mann_whitney", "anova", "kruskal")
METHOD_NAMES = {
    "welch": "Welch's t-test",
    "mann_whitney": "Mann-Whitney U",
    "anova": "Welch's ANOVA",
    "kruskal": "Kruskal-Wallis",
}
MIN_GROUP_N = 3
MAX_GROUPS = 12
# Per-group n above which the sampling distribution of the mean is close to normal
# for moderately skewed data (central limit theorem)
CLT_MIN_N = 30
CLT_MAX_ABS_SKEW = 1.0
# Hodges-Lehmann uses all pairwise differences; skip it for very large groups
HODGES_LEHMANN_MAX_PAIRS = 1_000_000
MAX_POSTHOC_LINES = 15


def check_group_normality(values: pd.Series) -> dict:
    """
    Judge whether a group is normal enough for a parametric test.

    A group passes if Shapiro-Wilk does not reject normality, or if it is large
    enough (n >= 30) and only moderately skewed (|skew| <= 1) for the central
    limit theorem to make the mean approximately normal.
    """
    n = len(values)
    skew = float(stats.skew(values))
    sample = values.sample(n=min(5000, n), random_state=42)
    shapiro_p = float(stats.shapiro(sample).pvalue) if values.nunique() > 1 else 1.0

    if shapiro_p >= ALPHA:
        ok, reason = True, "approximately normal"
    elif n >= CLT_MIN_N and abs(skew) <= CLT_MAX_ABS_SKEW:
        ok, reason = True, f"non-normal, but n >= {CLT_MIN_N} with moderate skew (CLT applies)"
    else:
        ok, reason = False, "non-normal"

    return {"n": n, "skew": skew, "shapiro_p": shapiro_p, "parametric_ok": ok, "reason": reason}


def interpret_rank_biserial(r: float) -> str:
    r = abs(r)
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "small"
    if r < 0.5:
        return "medium"
    return "large"


def interpret_variance_explained(es: float) -> str:
    """Conventional thresholds for omega-squared / epsilon-squared."""
    if es < 0.01:
        return "negligible"
    if es < 0.06:
        return "small"
    if es < 0.14:
        return "medium"
    return "large"


def omega_squared(groups: list[pd.Series]) -> float:
    """Omega-squared: less biased than eta-squared for the share of variance explained."""
    all_values = pd.concat(groups)
    grand_mean = all_values.mean()
    n, k = len(all_values), len(groups)
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    ms_within = (ss_total - ss_between) / (n - k)
    return float(max(0.0, (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)))


def hodges_lehmann(a: pd.Series, b: pd.Series) -> float | None:
    """Median of all pairwise differences a_i - b_j: the shift estimate paired with Mann-Whitney."""
    if len(a) * len(b) > HODGES_LEHMANN_MAX_PAIRS:
        return None
    return float(np.median(np.subtract.outer(a.to_numpy(), b.to_numpy())))


def _mann_whitney(a: pd.Series, b: pd.Series, names: tuple[str, str], column: str) -> tuple[str, float]:
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial correlation: P(a > b) - P(a < b)
    r_rb = 2 * result.statistic / (len(a) * len(b)) - 1
    shift = hodges_lehmann(a, b)
    significant = result.pvalue < ALPHA

    lines = [
        f"Mann-Whitney U Results: {column}",
        f"Group '{names[0]}': n={len(a)}, median={a.median():.2f}, IQR={a.quantile(.25):.2f}-{a.quantile(.75):.2f}",
        f"Group '{names[1]}': n={len(b)}, median={b.median():.2f}, IQR={b.quantile(.25):.2f}-{b.quantile(.75):.2f}",
        "",
        f"U statistic: {result.statistic:.1f}",
        f"p-value: {result.pvalue:.4g}",
        f"Rank-biserial correlation: {r_rb:.3f} ({interpret_rank_biserial(r_rb)} effect)",
    ]
    if shift is not None:
        lines.append(f"Hodges-Lehmann shift ('{names[0]}' - '{names[1]}'): {shift:.2f}")
    lines += [
        "",
        f"Interpretation: The distributions {'differ significantly' if significant else 'do not differ significantly'} "
        f"at alpha={ALPHA}, with a {interpret_rank_biserial(r_rb)} effect. "
        "This compares whole distributions (stochastic dominance), not means - describe medians, not means.",
    ]
    return "\n".join(lines), float(result.pvalue)


def _posthoc(groups: dict[str, pd.Series], parametric: bool) -> str:
    """Pairwise comparisons with Holm correction."""
    rows = []
    for (name_a, a), (name_b, b) in itertools.combinations(groups.items(), 2):
        if parametric:
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            diff = f"mean diff = {a.mean() - b.mean():.2f}"
        else:
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            diff = f"median diff = {a.median() - b.median():.2f}"
        rows.append((name_a, name_b, diff, float(p)))

    adjusted = holm([r[3] for r in rows])
    ranked = sorted(zip(rows, adjusted), key=lambda x: x[1])
    test = "Welch's t-tests" if parametric else "Mann-Whitney U tests"

    lines = [f"POST-HOC: pairwise {test}, Holm-corrected ({len(rows)} pairs)"]
    for (name_a, name_b, diff, p), p_adj in ranked[:MAX_POSTHOC_LINES]:
        flag = "significant" if p_adj < ALPHA else "n.s."
        lines.append(f"  {name_a} vs {name_b}: {diff}, p_adj = {p_adj:.4g} ({flag})")
    if len(ranked) > MAX_POSTHOC_LINES:
        lines.append(f"  ... {len(ranked) - MAX_POSTHOC_LINES} more pairs, all with larger p-values")
    return "\n".join(lines)


def compare_groups(
    df: pd.DataFrame,
    numeric_column: str,
    group_column: str,
    method: str = "auto"
) -> InferenceResult:
    """
    Compare a numeric column across two or more groups, choosing the test from the data.

    Args:
        df: The dataframe
        numeric_column: The numeric outcome to compare
        group_column: The categorical column defining the groups
        method: 'auto' (choose from assumption checks), 'welch', 'mann_whitney',
                'anova' (Welch's ANOVA), or 'kruskal'

    Returns:
        Assumption checks, the selected test's results, and post-hoc comparisons,
        plus the omnibus p-value for session-wide multiple-testing correction
    """
    if method not in METHODS:
        return InferenceResult(f"Error: method must be one of {METHODS}, got '{method}'.")
    for col in (numeric_column, group_column):
        if col not in df.columns:
            return InferenceResult(f"Error: Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        return InferenceResult(f"Error: Column '{numeric_column}' is not numeric.")

    data = df[[numeric_column, group_column]].dropna()
    sizes = data[group_column].value_counts()
    small = sorted(map(str, sizes[sizes < MIN_GROUP_N].index))
    kept = sorted(sizes[sizes >= MIN_GROUP_N].index, key=str)

    if len(kept) < 2:
        return InferenceResult(
            f"Error: Need at least 2 groups with n >= {MIN_GROUP_N} in '{group_column}', found {len(kept)}."
        )
    if len(kept) > MAX_GROUPS:
        return InferenceResult(
            f"Error: '{group_column}' has {len(kept)} groups; group comparison supports at most {MAX_GROUPS}. "
            "Is this column really categorical? Consider grouping rare categories or using run_regression."
        )
    if method in ("welch", "mann_whitney") and len(kept) != 2:
        return InferenceResult(f"Error: method '{method}' needs exactly 2 groups, found {len(kept)}. Use 'anova' or 'kruskal'.")

    data = data[data[group_column].isin(kept)]
    groups = {str(g): data.loc[data[group_column] == g, numeric_column] for g in kept}

    # --- Assumption checks ---
    checks = {name: check_group_normality(values) for name, values in groups.items()}
    all_parametric = all(c["parametric_ok"] for c in checks.values())
    levene_p = float(stats.levene(*groups.values(), center="median").pvalue)

    lines = [f"GROUP COMPARISON: {numeric_column} by {group_column}", "", "ASSUMPTION CHECKS:"]
    for name, c in checks.items():
        g = groups[name]
        lines.append(
            f"  {name} (n={c['n']}): mean={g.mean():.2f}, median={g.median():.2f}, sd={g.std():.2f}, "
            f"skew={c['skew']:.2f}, Shapiro-Wilk p={c['shapiro_p']:.3g} -> {c['reason']}"
        )
    lines.append(
        f"  Equal variances (Brown-Forsythe): p={levene_p:.3g} -> "
        f"{'variances differ' if levene_p < ALPHA else 'no evidence variances differ'} "
        "(Welch tests are used either way, as they don't assume equal variances)"
    )
    if small:
        lines.append(f"  Excluded groups with n < {MIN_GROUP_N}: {', '.join(small)}")

    # --- Test selection ---
    if method == "auto":
        if len(groups) == 2:
            method = "welch" if all_parametric else "mann_whitney"
        else:
            method = "anova" if all_parametric else "kruskal"
        why = "all groups meet the normality check" if all_parametric else "at least one group fails the normality check"
        lines.append(f"\nTEST SELECTED: {METHOD_NAMES[method]} ({why})")
    else:
        lines.append(f"\nTEST SELECTED: {METHOD_NAMES[method]} (requested)")
        if method in ("welch", "anova") and not all_parametric:
            lines.append("  Warning: at least one group is non-normal; a rank-based test may be more appropriate.")
    lines.append("")

    hypothesis = f"{numeric_column} differs across {group_column} groups"
    names = list(groups)

    # --- Run the test ---
    if method == "welch":
        result = perform_ttest(data, numeric_column, group_column)
        lines.append(result.text)
        p_value = result.p_values[0].p_value
    elif method == "mann_whitney":
        text, p_value = _mann_whitney(groups[names[0]], groups[names[1]], (names[0], names[1]), numeric_column)
        lines.append(text)
    elif method == "anova":
        res = anova_oneway(
            data[numeric_column], data[group_column], use_var="unequal", welch_correction=True
        )
        p_value = float(res.pvalue)
        w2 = omega_squared(list(groups.values()))
        lines += [
            f"Welch's ANOVA: F({res.df[0]:.0f}, {res.df[1]:.1f}) = {res.statistic:.3f}, p = {p_value:.4g}",
            f"Omega-squared: {w2:.3f} ({interpret_variance_explained(w2)} effect - "
            f"{group_column} explains ~{w2 * 100:.1f}% of the variance in {numeric_column})",
        ]
    else:
        res = stats.kruskal(*groups.values())
        p_value = float(res.pvalue)
        n = len(data)
        eps2 = float(res.statistic / (n - 1))
        lines += [
            f"Kruskal-Wallis: H({len(groups) - 1}) = {res.statistic:.3f}, p = {p_value:.4g}",
            f"Epsilon-squared: {eps2:.3f} ({interpret_variance_explained(eps2)} effect)",
            "This compares distributions via ranks - describe medians, not means.",
        ]

    if method in ("anova", "kruskal"):
        significant = p_value < ALPHA
        lines.append(
            f"\nInterpretation: {numeric_column} {'differs significantly' if significant else 'does not differ significantly'} "
            f"across {group_column} groups at alpha={ALPHA}."
        )
        if significant:
            lines += ["", _posthoc(groups, parametric=(method == "anova"))]
        else:
            lines.append("No post-hoc tests run because the omnibus test is not significant.")

    return InferenceResult("\n".join(lines), [PValue(METHOD_NAMES[method], hypothesis, p_value)])

