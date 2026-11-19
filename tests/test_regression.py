"""Tests for OLS regression with diagnostics."""

import re

import numpy as np
import pandas as pd
import pytest

from agent.tools.regression import clean_term_name, fit_ols


@pytest.fixture
def planted() -> pd.DataFrame:
    """y = 3 + 2*x1 - 1.5*x2 + 5*[group=B] + noise, with known coefficients."""
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "group": rng.choice(["A", "B"], n),
    })
    df["y"] = 3 + 2 * df.x1 - 1.5 * df.x2 + 5 * (df.group == "B") + rng.normal(0, 1, n)
    return df


def coefficient(text: str, name: str) -> tuple[float, float, float]:
    """Parse estimate and CI for a coefficient line."""
    line = next(l for l in text.splitlines() if l.strip().startswith(f"{name}:"))
    est, low, high = re.search(r": (-?[\d,.]+) \(SE [\d,.]+, CI \[(-?[\d,.]+), (-?[\d,.]+)\]", line).groups()
    return tuple(float(v.replace(",", "")) for v in (est, low, high))


def test_recovers_planted_coefficients(planted):
    result, fit = fit_ols(planted, "y", ["x1", "x2", "group"])
    for name, truth in [("x1", 2), ("x2", -1.5), ("group[T.B]", 5), ("Intercept", 3)]:
        est, low, high = coefficient(result.text, name)
        assert low <= truth <= high, name


def test_estimates_match_least_squares(planted):
    _, fit = fit_ols(planted, "y", ["x1", "x2", "group"])
    X = np.column_stack([np.ones(len(planted)), planted.group == "B", planted.x1, planted.x2])
    beta = np.linalg.lstsq(X.astype(float), planted.y, rcond=None)[0]
    assert np.allclose(fit.params.to_numpy(), beta)


def test_registers_one_p_value_per_coefficient(planted):
    result, _ = fit_ols(planted, "y", ["x1", "x2", "group"])
    assert len(result.p_values) == 3
    assert all(pv.test == "OLS coefficient" for pv in result.p_values)


def test_reports_reference_category(planted):
    result, _ = fit_ols(planted, "y", ["x1", "group"])
    assert "Reference categories: group='A'" in result.text


def test_heteroscedasticity_switches_to_robust_errors():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 500)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, x)})  # noise grows with x
    result, _ = fit_ols(df, "y", ["x"])
    assert "HC3 heteroscedasticity-robust" in result.text


def test_homoscedastic_data_keeps_classical_errors(planted):
    result, _ = fit_ols(planted, "y", ["x1", "x2"])
    assert "classical (homoscedasticity not rejected)" in result.text


def test_nonlinearity_flagged_by_reset():
    rng = np.random.default_rng(5)
    x = rng.uniform(-3, 3, 300)
    df = pd.DataFrame({"x": x, "y": x ** 2 + rng.normal(0, 0.5, 300)})
    result, _ = fit_ols(df, "y", ["x"])
    assert "possible nonlinearity" in result.text


def test_collinearity_flagged_by_vif():
    rng = np.random.default_rng(5)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.05, 300), "c": rng.normal(size=300)})
    df["y"] = df.a + df.c + rng.normal(size=300)
    result, _ = fit_ols(df, "y", ["a", "b", "c"])
    assert "-> severe" in result.text


def test_small_sample_warning(planted):
    result, _ = fit_ols(planted.head(25), "y", ["x1", "x2", "group"])
    assert "rule of thumb" in result.text


def test_drops_missing_rows(planted):
    planted.loc[:9, "x1"] = np.nan
    result, fit = fit_ols(planted, "y", ["x1"])
    assert "10 rows with missing values dropped" in result.text
    assert int(fit.nobs) == len(planted) - 10


@pytest.mark.parametrize("target,predictors,message", [
    ("group", ["x1"], "not numeric"),
    ("y", [], "at least one predictor"),
    ("y", ["nope"], "Columns not found"),
    ("y", ["y"], "cannot also be a predictor"),
])
def test_validation(planted, target, predictors, message):
    result, fit = fit_ols(planted, target, predictors)
    assert message in result.text
    assert fit is None and result.p_values == []


def test_rejects_id_like_categorical(employees):
    employees["employee_id"] = employees["employee_id"].astype(str)
    result, _ = fit_ols(employees, "salary", ["employee_id"])
    assert "Is it an ID column?" in result.text


def test_clean_term_name():
    assert clean_term_name("C(Q('dept'))[T.Sales]") == "dept[T.Sales]"
    assert clean_term_name("Q('years_experience')") == "years_experience"
