"""Tests for the two-group t-test tool."""

import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agent.tools.ttest import cohens_d, hedges_g, interpret_effect_size, perform_ttest


def _extract(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    assert match, f"Pattern {pattern!r} not found in:\n{text}"
    return float(match.group(1))


class TestEffectSize:
    def test_cohens_d_known_value(self):
        # Means 3 and 5, both variances 2.5 -> d = -2 / sqrt(2.5)
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([3, 4, 5, 6, 7])
        assert cohens_d(a, b) == pytest.approx(-2 / np.sqrt(2.5))

    def test_cohens_d_weights_pooled_sd_by_sample_size(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 1, 10))
        b = pd.Series(rng.normal(1, 3, 200))
        n1, n2 = len(a), len(b)
        pooled_sd = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2))
        assert cohens_d(a, b) == pytest.approx((a.mean() - b.mean()) / pooled_sd)

    def test_cohens_d_zero_variance(self):
        assert cohens_d(pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0])) == 0.0

    def test_hedges_g_correction(self):
        # J = 1 - 3 / (4 * 10 - 9) for n1 + n2 = 10
        assert hedges_g(1.0, 5, 5) == pytest.approx(1 - 3 / 31)

    @pytest.mark.parametrize("d,label", [
        (0.1, "negligible"), (-0.3, "small"), (0.6, "medium"), (-1.2, "large"),
    ])
    def test_interpretation(self, d, label):
        assert interpret_effect_size(d) == label


class TestPerformTtest:
    def test_defaults_to_welch(self, two_groups):
        output = perform_ttest(two_groups, "value", "group").text
        a = two_groups.loc[two_groups.group == "A", "value"]
        b = two_groups.loc[two_groups.group == "B", "value"]
        welch = stats.ttest_ind(a, b, equal_var=False)
        student = stats.ttest_ind(a, b, equal_var=True)

        assert "Welch" in output
        assert _extract(r"t-statistic: (-?[\d.]+)", output) == pytest.approx(welch.statistic, abs=1e-3)
        # Unequal variances and sizes make the two tests diverge - make sure we didn't use Student's
        assert abs(welch.statistic - student.statistic) > 0.1

    def test_confidence_interval_matches_welch_satterthwaite(self, two_groups):
        output = perform_ttest(two_groups, "value", "group").text
        a = two_groups.loc[two_groups.group == "A", "value"]
        b = two_groups.loc[two_groups.group == "B", "value"]

        va, vb = a.var() / len(a), b.var() / len(b)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        margin = stats.t.ppf(0.975, dof) * np.sqrt(va + vb)
        diff = a.mean() - b.mean()

        match = re.search(r"CI for the difference: \[(-?[\d.]+), (-?[\d.]+)\]", output)
        assert match
        assert float(match.group(1)) == pytest.approx(diff - margin, abs=0.01)
        assert float(match.group(2)) == pytest.approx(diff + margin, abs=0.01)
        assert _extract(r"df = ([\d.]+)", output) == pytest.approx(dof, abs=0.1)

    def test_student_option(self, two_groups):
        output = perform_ttest(two_groups, "value", "group", equal_var=True).text
        assert "Student" in output

    def test_group_order_is_deterministic(self):
        df = pd.DataFrame({"v": [10.0, 11, 12, 1, 2, 3], "g": ["z", "z", "z", "a", "a", "a"]})
        output = perform_ttest(df, "v", "g").text
        # Groups are sorted, so the difference is 'a' - 'z' regardless of row order
        assert "Mean difference ('a' - 'z'): -9.00" in output

    def test_no_difference_is_not_significant(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "v": rng.normal(0, 1, 200),
            "g": ["x", "y"] * 100,
        })
        output = perform_ttest(df, "v", "g").text
        assert "not statistically significant" in output
        assert "CI includes 0" in output

    def test_requires_two_groups(self, employees):
        assert "exactly 2 groups" in perform_ttest(employees, "salary", "department").text

    def test_requires_two_observations_per_group(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["a", "a", "b"]})
        assert "at least 2 observations" in perform_ttest(df, "v", "g").text

    def test_rejects_non_numeric_column(self, employees):
        assert "not numeric" in perform_ttest(employees, "department", "gender").text

    def test_missing_column(self, employees):
        assert "not found" in perform_ttest(employees, "nope", "gender").text

    def test_returns_p_value_for_registry(self, two_groups):
        result = perform_ttest(two_groups, "value", "group")
        a = two_groups.loc[two_groups.group == "A", "value"]
        b = two_groups.loc[two_groups.group == "B", "value"]
        [pv] = result.p_values
        assert pv.test == "Welch's t-test"
        assert pv.p_value == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)

    def test_errors_produce_no_p_values(self, employees):
        assert perform_ttest(employees, "salary", "department").p_values == []
