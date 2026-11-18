"""Tests for the normality test tool."""

import numpy as np
import pandas as pd

from agent.tools.normality import check_normality


def test_normal_data_appears_normal():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    assert "appears to be normally distributed" in check_normality(df, "x")


def test_skewed_data_is_not_normal():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.exponential(size=200)})
    assert "does NOT appear to be normally distributed" in check_normality(df, "x")


def test_large_sample_caveat():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=8000)})
    output = check_normality(df, "x")
    assert "subsample of 5,000 of 8,000" in output
    assert "detects even trivial departures" in output


def test_small_sample_caveat():
    df = pd.DataFrame({"x": [1.0, 2.5, 2.0, 3.1, 4.0, 2.2, 3.3, 1.8]})
    assert "little power" in check_normality(df, "x")


def test_rejects_non_numeric():
    df = pd.DataFrame({"x": list("abcde")})
    assert "not numeric" in check_normality(df, "x")
