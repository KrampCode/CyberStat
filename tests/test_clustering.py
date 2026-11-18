"""Tests for K-means clustering and cluster-count selection."""

import numpy as np
import pandas as pd
import pytest

from agent.tools.derived import get_derived_columns
from agent.tools.kmeans import elbow_k, find_optimal_k, interpret_silhouette, perform_kmeans


class TestElbow:
    def test_elbow_at_knee(self):
        # Normalized gaps below the chord at k=3,4,5 are 0.19, 0.48, 0.24: the knee is k=4.
        # (The previous second-difference implementation had an off-by-one and returned k=5.)
        k_values = [2, 3, 4, 5, 6]
        inertias = [1000, 600, 100, 90, 85]
        assert elbow_k(k_values, inertias) == 4

    def test_not_fooled_by_large_first_drop(self):
        # A raw second difference picks k=3 here because the 2->3 drop dominates in absolute terms
        k_values = [2, 3, 4, 5, 6, 7, 8]
        inertias = [584.8, 222.3, 78.6, 68.2, 58.2, 48.3, 41.8]
        assert elbow_k(k_values, inertias) == 4

    def test_needs_three_points(self):
        assert elbow_k([2, 3], [100, 50]) is None

    def test_flat_curve(self):
        assert elbow_k([2, 3, 4], [10, 10, 10]) is None


class TestFindOptimalK:
    def test_recovers_true_k(self, blobs):
        result = find_optimal_k(blobs, max_k=8)
        assert result["suggested_k"] == 4
        assert result["elbow_k"] == 4
        assert result["best_silhouette"] > 0.5
        assert result["features_used"] == ["x1", "x2", "x3"]

    def test_reports_one_score_per_k(self, blobs):
        result = find_optimal_k(blobs, max_k=6)
        assert result["k_values"] == [2, 3, 4, 5, 6]
        assert len(result["inertias"]) == len(result["silhouettes"]) == 5

    def test_unstructured_data_has_low_silhouette(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.uniform(size=(300, 3)), columns=["a", "b", "c"])
        result = find_optimal_k(df, max_k=6)
        assert result["best_silhouette"] < 0.5

    def test_too_few_rows(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 4 complete rows"):
            find_optimal_k(df)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"a": list("abcdef")})
        with pytest.raises(ValueError, match="numeric column"):
            find_optimal_k(df)


class TestPerformKmeans:
    def test_clusters_match_true_labels(self, blobs):
        result_df, stats = perform_kmeans(blobs, n_clusters=4)
        # Each true blob should map to exactly one cluster
        crosstab = pd.crosstab(result_df["true_label"], result_df["cluster"])
        assert ((crosstab > 0).sum(axis=1) == 1).all()
        assert stats["silhouette"] > 0.5
        assert stats["silhouette_interpretation"] in {"strong cluster structure", "reasonable cluster structure"}

    def test_marks_cluster_column_as_derived(self, blobs):
        result_df, _ = perform_kmeans(blobs, n_clusters=4)
        assert "cluster" in get_derived_columns(result_df)
        assert str(result_df["cluster"].dtype) == "Int64"

    def test_rows_with_missing_values_get_no_cluster(self, blobs):
        blobs.loc[0, "x1"] = np.nan
        result_df, stats = perform_kmeans(blobs, n_clusters=4)
        assert pd.isna(result_df.loc[0, "cluster"])
        assert stats["n_samples"] == len(blobs) - 1

    def test_does_not_mutate_input(self, blobs):
        original_columns = list(blobs.columns)
        perform_kmeans(blobs, n_clusters=4)
        assert list(blobs.columns) == original_columns

    def test_rejects_single_cluster(self, blobs):
        with pytest.raises(ValueError, match="at least 2"):
            perform_kmeans(blobs, n_clusters=1)


@pytest.mark.parametrize("score,label", [
    (0.8, "strong cluster structure"),
    (0.6, "reasonable cluster structure"),
    (0.3, "weak cluster structure - clusters may be artificial"),
    (0.1, "no substantial cluster structure"),
])
def test_interpret_silhouette(score, label):
    assert interpret_silhouette(score) == label
