"""Tests for assumption-aware group comparison."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agent.tools.group_comparison import compare_groups, omega_squared


def make_groups(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({
        "value": np.concatenate(list(samples.values())),
        "group": np.repeat(list(samples), [len(s) for s in samples.values()]),
    })


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def welch_anova_by_hand(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Welch's F and degrees of freedom from the closed-form formulas."""
    k = len(groups)
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    w = n / np.array([g.var(ddof=1) for g in groups])
    weighted_mean = (w * means).sum() / w.sum()
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = ((w * (means - weighted_mean) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    return f, k - 1, (k ** 2 - 1) / (3 * lam)


class TestTestSelection:
    def test_two_normal_groups_use_welch(self, rng):
        df = make_groups({"a": rng.normal(0, 1, 40), "b": rng.normal(0.8, 2, 50)})
        result = compare_groups(df, "value", "group")
        assert "TEST SELECTED: Welch's t-test" in result.text
        assert result.p_values[0].test == "Welch's t-test"

    def test_small_skewed_groups_use_mann_whitney(self, rng):
        df = make_groups({"a": rng.lognormal(0, 1.2, 20), "b": rng.lognormal(1, 1.2, 20)})
        result = compare_groups(df, "value", "group")
        assert "TEST SELECTED: Mann-Whitney U" in result.text
        a, b = df[df.group == "a"].value, df[df.group == "b"].value
        assert result.p_values[0].p_value == pytest.approx(stats.mannwhitneyu(a, b).pvalue)

    def test_large_moderately_skewed_groups_stay_parametric(self, rng):
        # Gamma(shape=9) has skew 2/3: Shapiro rejects at n=200, but the CLT makes the t-test valid
        df = make_groups({"a": rng.gamma(9, 1, 200), "b": rng.gamma(9, 1.1, 200)})
        result = compare_groups(df, "value", "group")
        assert "CLT applies" in result.text
        assert "TEST SELECTED: Welch's t-test" in result.text

    def test_three_normal_groups_use_welch_anova(self, rng):
        groups = [rng.normal(10, 1, 30), rng.normal(10, 3, 45), rng.normal(12, 2, 25)]
        df = make_groups(dict(zip("abc", groups)))
        result = compare_groups(df, "value", "group")
        assert "TEST SELECTED: Welch's ANOVA" in result.text
        f, df1, df2 = welch_anova_by_hand(groups)
        assert f"F({df1}, {df2:.1f}) = {f:.3f}" in result.text
        assert result.p_values[0].p_value == pytest.approx(stats.f.sf(f, df1, df2))

    def test_three_skewed_groups_use_kruskal(self, rng):
        groups = [rng.exponential(1, 15), rng.exponential(2, 15), rng.exponential(4, 15)]
        df = make_groups(dict(zip("abc", groups)))
        result = compare_groups(df, "value", "group")
        assert "TEST SELECTED: Kruskal-Wallis" in result.text
        assert result.p_values[0].p_value == pytest.approx(stats.kruskal(*groups).pvalue)

    def test_forced_method_warns_when_assumptions_fail(self, rng):
        df = make_groups({"a": rng.lognormal(0, 1.2, 20), "b": rng.lognormal(1, 1.2, 20)})
        result = compare_groups(df, "value", "group", method="welch")
        assert "(requested)" in result.text
        assert "rank-based test may be more appropriate" in result.text


class TestPostHoc:
    def test_identifies_the_shifted_group(self, rng):
        df = make_groups({
            "a": rng.normal(0, 1, 50),
            "b": rng.normal(0, 1, 50),
            "c": rng.normal(1.5, 1, 50),
        })
        text = compare_groups(df, "value", "group").text
        assert "Holm-corrected (3 pairs)" in text
        assert "a vs c" in text and "b vs c" in text
        posthoc = {line.split(":")[0].strip(): line for line in text.split("POST-HOC")[1].splitlines()[1:]}
        assert "(significant)" in posthoc["a vs c"]
        assert "(significant)" in posthoc["b vs c"]
        assert "(n.s.)" in posthoc["a vs b"]

    def test_skipped_when_omnibus_not_significant(self, rng):
        df = make_groups({g: rng.normal(0, 1, 40) for g in "abc"})
        text = compare_groups(df, "value", "group").text
        assert "No post-hoc tests run" in text
        assert "POST-HOC" not in text


class TestEffectSizes:
    def test_omega_squared_near_zero_without_effect(self, rng):
        groups = [pd.Series(rng.normal(0, 1, 500)) for _ in range(3)]
        assert omega_squared(groups) < 0.01

    def test_omega_squared_large_with_big_shift(self, rng):
        groups = [pd.Series(rng.normal(mu, 1, 100)) for mu in (0, 2, 4)]
        assert omega_squared(groups) > 0.6

    def test_rank_biserial_is_one_for_complete_separation(self):
        df = make_groups({"a": np.arange(10.0, 20.0), "b": np.arange(0.0, 10.0)})
        text = compare_groups(df, "value", "group", method="mann_whitney").text
        assert "Rank-biserial correlation: 1.000" in text


class TestValidation:
    def test_excludes_tiny_groups(self, rng):
        df = make_groups({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30), "c": np.array([1.0, 2.0])})
        result = compare_groups(df, "value", "group")
        assert "Excluded groups with n < 3: c" in result.text
        assert "Welch's t-test" in result.text

    def test_too_many_groups(self, rng):
        df = pd.DataFrame({"value": rng.normal(size=300), "group": np.repeat([f"g{i}" for i in range(15)], 20)})
        assert "at most 12" in compare_groups(df, "value", "group").text

    def test_two_group_method_needs_two_groups(self, employees):
        assert "needs exactly 2 groups" in compare_groups(employees, "salary", "department", method="welch").text

    def test_unknown_method(self, employees):
        assert "method must be one of" in compare_groups(employees, "salary", "gender", method="ttest").text

    def test_errors_produce_no_p_values(self, employees):
        assert compare_groups(employees, "department", "gender").p_values == []
