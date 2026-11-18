"""K-means clustering tool."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .derived import mark_derived, select_numeric

# Silhouette is O(n^2); score on a fixed random sample for large datasets
SILHOUETTE_SAMPLE_SIZE = 2000


def _silhouette(scaled_data: np.ndarray, labels: np.ndarray, random_state: int) -> float:
    """Mean silhouette coefficient, sampled for large datasets."""
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(scaled_data))
    return float(silhouette_score(
        scaled_data, labels, sample_size=sample_size, random_state=random_state
    ))


def interpret_silhouette(score: float) -> str:
    """Rule-of-thumb reading of a mean silhouette coefficient (Kaufman & Rousseeuw)."""
    if score >= 0.7:
        return "strong cluster structure"
    if score >= 0.5:
        return "reasonable cluster structure"
    if score >= 0.25:
        return "weak cluster structure - clusters may be artificial"
    return "no substantial cluster structure"


def elbow_k(k_values: list[int], inertias: list[float]) -> int | None:
    """
    Locate the elbow of an inertia curve with the Kneedle method (Satopaa et al., 2011).

    Both axes are normalized to [0, 1], and the elbow is the point furthest below the
    chord joining the first and last points. Unlike the raw second difference, this is
    scale-invariant and not biased toward small k by a large first drop in inertia.
    Returns None if there are fewer than 3 points or the curve is flat.
    """
    if len(inertias) < 3 or inertias[0] == inertias[-1]:
        return None
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y[-1]) / (y[0] - y[-1])
    # For a decreasing curve the normalized chord is y = 1 - x
    gap = (1 - x_norm) - y_norm
    return int(k_values[int(np.argmax(gap))])


def perform_kmeans(
    df: pd.DataFrame,
    columns: list[str] = None,
    n_clusters: int = 3,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Perform K-means clustering on numeric columns.
    
    Args:
        df: The dataframe
        columns: List of columns to use for clustering (None = all numeric)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (df with cluster column, statistics_dict)
    """
    numeric_df = select_numeric(df, columns)
    
    if len(numeric_df.columns) < 1:
        raise ValueError("Need at least 1 numeric column for clustering")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    
    # Drop rows with missing values
    clean_df = numeric_df.dropna()
    if len(clean_df) < n_clusters * 2:
        raise ValueError(f"Need at least {n_clusters * 2} complete rows for {n_clusters} clusters")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to original dataframe
    result_df = df.copy()
    result_df['cluster'] = np.nan
    result_df.loc[clean_df.index, 'cluster'] = labels
    result_df['cluster'] = result_df['cluster'].astype('Int64')  # Nullable integer
    mark_derived(result_df, ['cluster'])
    
    silhouette = _silhouette(scaled_data, labels, random_state)
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_data = clean_df[mask]
        
        # Per-feature statistics for this cluster
        feature_stats = {}
        for col in clean_df.columns:
            feature_stats[col] = {
                "mean": round(float(cluster_data[col].mean()), 2),
                "std": round(float(cluster_data[col].std()), 2),
                "min": round(float(cluster_data[col].min()), 2),
                "max": round(float(cluster_data[col].max()), 2),
            }
        
        cluster_stats[f"Cluster {cluster_id}"] = {
            "size": int(mask.sum()),
            "percentage": round(mask.sum() / len(labels) * 100, 1),
            "feature_means": {col: round(float(cluster_data[col].mean()), 2) for col in clean_df.columns},
            "feature_stats": feature_stats
        }
    
    # Identify distinguishing features for each cluster
    global_means = clean_df.mean()
    global_stds = clean_df.std()
    
    distinguishing_features = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_data = clean_df[mask]
        cluster_means = cluster_data.mean()
        
        # Z-score of cluster mean vs global mean
        z_scores = (cluster_means - global_means) / global_stds
        top_features = z_scores.abs().sort_values(ascending=False).head(3)
        
        distinguishing_features[f"Cluster {cluster_id}"] = [
            {
                "feature": feat,
                "direction": "high" if z_scores[feat] > 0 else "low",
                "cluster_mean": round(float(cluster_means[feat]), 2),
                "global_mean": round(float(global_means[feat]), 2)
            }
            for feat in top_features.index
        ]
    
    stats = {
        "n_samples": len(clean_df),
        "n_clusters": n_clusters,
        "features_used": list(clean_df.columns),
        "inertia": round(float(kmeans.inertia_), 2),
        "silhouette": round(silhouette, 3),
        "silhouette_interpretation": interpret_silhouette(silhouette),
        "cluster_stats": cluster_stats,
        "distinguishing_features": distinguishing_features,
    }
    
    return result_df, stats


def find_optimal_k(
    df: pd.DataFrame,
    columns: list[str] = None,
    max_k: int = 10,
    random_state: int = 42
) -> dict:
    """
    Compare cluster counts using silhouette scores and the elbow method.
    
    Args:
        df: The dataframe
        columns: List of columns to use (None = all numeric)
        max_k: Maximum number of clusters to try
        random_state: Random seed
        
    Returns:
        Dictionary with inertia and silhouette per k, the silhouette-optimal k
        (suggested_k), and the elbow k for comparison
    """
    numeric_df = select_numeric(df, columns)
    if len(numeric_df.columns) < 1:
        raise ValueError("Need at least 1 numeric column for clustering")
    
    # Drop rows with missing values
    clean_df = numeric_df.dropna()
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df)
    
    # Try different k values (silhouette needs k <= n_samples - 1)
    max_k = min(max_k, len(clean_df) // 2)
    if max_k < 2:
        raise ValueError(f"Need at least 4 complete rows to compare cluster counts, found {len(clean_df)}")
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        inertias.append(float(kmeans.inertia_))
        silhouettes.append(_silhouette(scaled_data, labels, random_state))
    
    # Silhouette is the primary criterion; the elbow is reported for comparison
    silhouette_k = k_range[int(np.argmax(silhouettes))]
    best_silhouette = max(silhouettes)
    
    return {
        "k_values": k_range,
        "inertias": [round(i, 2) for i in inertias],
        "silhouettes": [round(sc, 3) for sc in silhouettes],
        "suggested_k": silhouette_k,
        "elbow_k": elbow_k(k_range, inertias),
        "best_silhouette": round(best_silhouette, 3),
        "silhouette_interpretation": interpret_silhouette(best_silhouette),
        "n_samples": len(clean_df),
        "features_used": list(clean_df.columns)
    }
