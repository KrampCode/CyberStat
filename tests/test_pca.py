"""Tests for the PCA tool."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from agent.tools.derived import get_derived_columns
from agent.tools.pca import perform_pca

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_explained_variance_matches_sklearn(iris):
    _, stats = perform_pca(iris)
    reference = PCA().fit(StandardScaler().fit_transform(iris[IRIS_FEATURES]))
    for i, ratio in enumerate(reference.explained_variance_ratio_):
        assert stats["explained_variance"][f"PC{i+1}"]["variance"] == pytest.approx(ratio, abs=1e-4)


def test_cumulative_variance_thresholds(iris):
    _, stats = perform_pca(iris)
    cumulative = [v["cumulative"] for v in stats["explained_variance"].values()]
    k90 = stats["components_for_90_variance"]
    assert cumulative[k90 - 1] >= 90
    assert k90 == 1 or cumulative[k90 - 2] < 90


def test_uses_only_numeric_source_columns(iris):
    _, stats = perform_pca(iris)
    assert stats["features_used"] == IRIS_FEATURES


def test_marks_components_as_derived(iris):
    result_df, _ = perform_pca(iris, n_components=2)
    assert get_derived_columns(result_df) == ["PC1", "PC2"]


def test_rerun_replaces_previous_components(iris):
    first, _ = perform_pca(iris)  # PC1..PC4
    second, stats = perform_pca(first, n_components=2)
    assert [c for c in second.columns if c.startswith("PC")] == ["PC1", "PC2"]
    # The rerun must fit on the original features, not on the previous components
    assert stats["features_used"] == IRIS_FEATURES


def test_rows_with_missing_values_get_no_components(iris):
    iris.loc[0, "sepal_length"] = np.nan
    result_df, stats = perform_pca(iris)
    assert np.isnan(result_df.loc[0, "PC1"])
    assert stats["n_samples"] == len(iris) - 1


def test_requires_two_columns(iris):
    with pytest.raises(ValueError, match="at least 2 numeric columns"):
        perform_pca(iris, columns=["sepal_length"])
