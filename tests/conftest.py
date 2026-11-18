"""Shared fixtures for CyberStat tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def iris() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "iris.csv")


@pytest.fixture
def employees() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "employee_data.csv")


@pytest.fixture
def blobs() -> pd.DataFrame:
    """Four well-separated clusters in three dimensions - the true k is 4."""
    X, y = make_blobs(n_samples=400, centers=4, n_features=3, cluster_std=0.6, random_state=0)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["true_label"] = y.astype(str)
    return df


@pytest.fixture
def two_groups() -> pd.DataFrame:
    """Two groups with unequal sizes and variances - the case where Welch and Student differ."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=50, scale=5, size=30)
    b = rng.normal(loc=55, scale=15, size=120)
    return pd.DataFrame({
        "value": np.concatenate([a, b]),
        "group": ["A"] * len(a) + ["B"] * len(b),
    })
