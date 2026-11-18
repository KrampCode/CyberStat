"""
Regression tests: columns created by PCA/K-means must not contaminate later analyses.

Previously, running PCA added PC1..PCn to the working dataframe, and every tool that
defaults to "all numeric columns" (correlations, describe, heatmap, a second
clustering run) silently included them.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from agent.tools import (
    calculate_correlation,
    create_correlation_heatmap,
    describe_dataframe,
    find_optimal_k,
    perform_kmeans,
    perform_pca,
)
from agent.tools.derived import get_derived_columns, mark_derived, select_numeric

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@pytest.fixture
def analyzed_iris(iris):
    """Iris after the agent's PCA -> K-means workflow."""
    df, _ = perform_pca(iris)
    df, _ = perform_kmeans(df, n_clusters=3)
    return df


def test_derived_columns_survive_both_steps(analyzed_iris):
    assert set(get_derived_columns(analyzed_iris)) == {"PC1", "PC2", "PC3", "PC4", "cluster"}


def test_clustering_after_pca_uses_source_features(analyzed_iris):
    assert find_optimal_k(analyzed_iris, max_k=4)["features_used"] == IRIS_FEATURES


def test_correlations_exclude_derived(analyzed_iris):
    output = calculate_correlation(analyzed_iris)
    assert "PC1" not in output and "cluster" not in output


def test_heatmap_excludes_derived(analyzed_iris):
    _, _, stats = create_correlation_heatmap(analyzed_iris)
    assert stats["n_variables"] == 4


def test_describe_excludes_derived(analyzed_iris):
    output = describe_dataframe(analyzed_iris)
    assert "PC1" not in output and "cluster" not in output


def test_explicit_columns_are_honored(analyzed_iris):
    assert list(select_numeric(analyzed_iris, ["PC1", "PC2"]).columns) == ["PC1", "PC2"]


def test_explicit_missing_column_raises(iris):
    with pytest.raises(ValueError, match="Columns not found"):
        select_numeric(iris, ["nope"])


def test_mark_derived_ignores_dropped_columns():
    df = pd.DataFrame({"a": [1], "PC1": [0.5]})
    mark_derived(df, ["PC1"])
    assert get_derived_columns(df.drop(columns=["PC1"])) == []
