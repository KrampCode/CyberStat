"""K-means clustering tool."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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
    # Select columns
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 1:
        raise ValueError("Need at least 1 numeric column for clustering")
    
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
    Use elbow method to suggest optimal number of clusters.
    
    Args:
        df: The dataframe
        columns: List of columns to use (None = all numeric)
        max_k: Maximum number of clusters to try
        random_state: Random seed
        
    Returns:
        Dictionary with inertia values and suggested k
    """
    # Select columns
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    # Drop rows with missing values
    clean_df = numeric_df.dropna()
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df)
    
    # Try different k values
    max_k = min(max_k, len(clean_df) // 2)
    k_range = range(2, max_k + 1)
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Find elbow using the "knee" method (largest second derivative)
    if len(inertias) >= 3:
        # Calculate rate of change
        diffs = np.diff(inertias)
        # Calculate second derivative (change in rate of change)
        second_diffs = np.diff(diffs)
        # Elbow is where second derivative is maximum (curve bends most)
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because we lost 2 points from diffs
        suggested_k = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else 3
    else:
        suggested_k = 3
    
    return {
        "k_values": list(k_range),
        "inertias": [round(float(i), 2) for i in inertias],
        "suggested_k": suggested_k,
        "n_samples": len(clean_df),
        "features_used": list(clean_df.columns)
    }
