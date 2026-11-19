"""Tests for the chi-square / Fisher's exact test of independence."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agent.tools.independence import independence_test, interpret_cramers_v


def from_counts(counts: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows = [(a, b) for (a, b), n in counts.items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["x", "y"])


def test_matches_scipy_chi_square():
    df = from_counts({("a", "p"): 30, ("a", "q"): 20, ("b", "p"): 15, ("b", "q"): 35, ("c", "p"): 25, ("c", "q"): 25})
    result = independence_test(df, "x", "y")
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.x, df.y))
    assert f"Chi-square({dof}) = {chi2:.3f}" in result.text
    assert result.p_values[0].p_value == pytest.approx(p)


def test_cramers_v_by_hand():
    df = from_counts({("a", "p"): 30, ("a", "q"): 20, ("b", "p"): 15, ("b", "q"): 35, ("c", "p"): 25, ("c", "q"): 25})
    table = pd.crosstab(df.x, df.y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    v = np.sqrt(chi2 / (len(df) * (min(table.shape) - 1)))
    assert f"Cramer's V: {v:.3f}" in independence_test(df, "x", "y").text


def test_fisher_for_small_2x2():
    df = from_counts({("a", "p"): 4, ("a", "q"): 1, ("b", "p"): 0, ("b", "q"): 5})
    result = independence_test(df, "x", "y")
    assert "Fisher's exact test" in result.text
    assert result.p_values[0].p_value == pytest.approx(stats.fisher_exact([[4, 1], [0, 5]])[1])


def test_warns_on_sparse_tables():
    df = from_counts({("a", "p"): 50, ("a", "q"): 2, ("a", "r"): 1, ("b", "p"): 40, ("b", "q"): 1, ("b", "r"): 3})
    assert "chi-square approximation is unreliable" in independence_test(df, "x", "y").text


def test_identifies_cells_driving_association():
    # Planted: segment "b" strongly prefers channel "q"
    df = from_counts({
        ("a", "p"): 40, ("a", "q"): 40, ("a", "r"): 40,
        ("b", "p"): 20, ("b", "q"): 80, ("b", "r"): 20,
    })
    text = independence_test(df, "x", "y").text
    drivers = text.split("CELLS DRIVING THE ASSOCIATION")[1].splitlines()[1:]
    assert any("x=b, y=q" in line and "more than expected" in line for line in drivers)
    assert any("x=a, y=q" in line and "fewer than expected" in line for line in drivers)


def test_no_association():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.choice(list("abc"), 600), "y": rng.choice(list("pq"), 600)})
    result = independence_test(df, "x", "y")
    assert "not significantly associated" in result.text
    assert "CELLS DRIVING" not in result.text


def test_rejects_high_cardinality(employees):
    assert "at most 20 levels" in independence_test(employees, "employee_id", "gender").text


def test_rejects_same_column(employees):
    assert "two different columns" in independence_test(employees, "gender", "gender").text


@pytest.mark.parametrize("v,min_dim,label", [
    (0.25, 2, "small"), (0.25, 3, "medium"), (0.25, 5, "medium"), (0.05, 2, "negligible"), (0.6, 2, "large"),
])
def test_cramers_v_thresholds_depend_on_table_size(v, min_dim, label):
    assert interpret_cramers_v(v, min_dim) == label
