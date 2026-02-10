"""Cluster visualization tools."""

import io
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import THEME, CYBER_PALETTE, format_label


def create_cluster_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    cluster_column: str = 'cluster',
    show_centroids: bool = True
) -> tuple[bytes, str, dict]:
    """
    Create a scatter plot colored by cluster assignments.
    
    Args:
        df: Dataframe with cluster assignments
        x_column: Column for x-axis
        y_column: Column for y-axis
        cluster_column: Column containing cluster labels
        show_centroids: Whether to mark cluster centroids
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    if cluster_column not in df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found")
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"Columns not found")
    
    # Get clean data
    plot_df = df[[x_column, y_column, cluster_column]].dropna()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    clusters = sorted(plot_df[cluster_column].unique())
    
    centroids = {}
    for i, cluster in enumerate(clusters):
        color = CYBER_PALETTE[i % len(CYBER_PALETTE)]
        mask = plot_df[cluster_column] == cluster
        cluster_data = plot_df[mask]
        
        ax.scatter(
            cluster_data[x_column],
            cluster_data[y_column],
            c=[color],
            label=f'Cluster {int(cluster)}',
            alpha=0.7,
            s=50,
            edgecolors='none'
        )
        
        # Calculate and store centroid
        centroid_x = cluster_data[x_column].mean()
        centroid_y = cluster_data[y_column].mean()
        centroids[f"Cluster {int(cluster)}"] = {
            "x": round(float(centroid_x), 2),
            "y": round(float(centroid_y), 2),
            "size": len(cluster_data)
        }
        
        if show_centroids:
            ax.scatter(
                centroid_x, centroid_y,
                c=[color],
                marker='X',
                s=200,
                edgecolors=THEME['text'],
                linewidths=2
            )
    
    ax.set_xlabel(format_label(x_column), color='#ffffff', fontweight='medium')
    ax.set_ylabel(format_label(y_column), color='#ffffff', fontweight='medium')
    ax.set_title('Cluster Analysis')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    
    # Add subtle crosshairs at origin if data spans it
    if plot_df[x_column].min() < 0 < plot_df[x_column].max():
        ax.axvline(x=0, color=THEME['text_secondary'], linestyle='--', alpha=0.4, linewidth=1)
    if plot_df[y_column].min() < 0 < plot_df[y_column].max():
        ax.axhline(y=0, color=THEME['text_secondary'], linestyle='--', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    stats = {
        "n_points": len(plot_df),
        "n_clusters": len(clusters),
        "x_column": x_column,
        "y_column": y_column,
        "centroids": centroids,
        "cluster_sizes": {f"Cluster {int(c)}": int((plot_df[cluster_column] == c).sum()) for c in clusters}
    }
    
    return buf.getvalue(), f"Cluster scatter plot ({format_label(y_column)} vs {format_label(x_column)})", stats


def create_elbow_plot(k_values: list, inertias: list, suggested_k: int) -> tuple[bytes, str, dict]:
    """
    Create an elbow plot for choosing optimal number of clusters.
    
    Args:
        k_values: List of k values tried
        inertias: List of inertia values for each k
        suggested_k: The suggested optimal k
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(k_values, inertias, color=THEME['neon_blue'], marker='o', linewidth=2, 
           markersize=8, markerfacecolor=THEME['neon_blue'])
    
    # Mark the suggested elbow point
    if suggested_k in k_values:
        idx = k_values.index(suggested_k)
        ax.axvline(x=suggested_k, color=THEME['neon_pink'], linestyle='--', alpha=0.8, 
                   linewidth=2, label=f'Suggested k={suggested_k}')
        ax.scatter([suggested_k], [inertias[idx]], c=THEME['neon_pink'], s=200, 
                  zorder=5, marker='*', edgecolors=THEME['text'])
    
    ax.set_xlabel('Number of Clusters (k)', color='#ffffff', fontweight='medium')
    ax.set_ylabel('Inertia', color='#ffffff', fontweight='medium')
    ax.set_title('Elbow Method')
    ax.set_xticks(k_values)
    ax.legend(framealpha=0.9)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    stats = {
        "k_range": [min(k_values), max(k_values)],
        "suggested_k": suggested_k,
        "inertia_at_suggested_k": inertias[k_values.index(suggested_k)] if suggested_k in k_values else None,
        "inertia_reduction": {
            f"k={k_values[i]}->{k_values[i+1]}": round(inertias[i] - inertias[i+1], 2)
            for i in range(len(k_values)-1)
        }
    }
    
    return buf.getvalue(), "Elbow plot for selecting optimal number of clusters", stats


def create_cluster_profile_chart(cluster_stats: dict, features: list = None) -> tuple[bytes, str, dict]:
    """
    Create a bar chart showing feature means by cluster (cluster profiles).
    
    Args:
        cluster_stats: Cluster statistics dict from kmeans
        features: List of features to show (None = all)
        
    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    clusters = list(cluster_stats.keys())
    
    # Get feature names from first cluster
    all_features = list(cluster_stats[clusters[0]]["feature_means"].keys())
    if features:
        all_features = [f for f in features if f in all_features]
    
    # Limit to 6 features for readability
    if len(all_features) > 6:
        all_features = all_features[:6]
    
    # Build data for plotting
    x = np.arange(len(all_features))
    width = 0.8 / len(clusters)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for i, cluster in enumerate(clusters):
        color = CYBER_PALETTE[i % len(CYBER_PALETTE)]
        means = [cluster_stats[cluster]["feature_means"][f] for f in all_features]
        offset = width * i - width * (len(clusters) - 1) / 2
        ax.bar(x + offset, means, width, label=cluster, color=color, alpha=0.85,
              edgecolor=THEME['plot_bg'])
    
    ax.set_xlabel('Features', color='#ffffff', fontweight='medium')
    ax.set_ylabel('Mean Value', color='#ffffff', fontweight='medium')
    ax.set_title('Cluster Profiles')
    ax.set_xticks(x)
    ax.set_xticklabels([format_label(f) for f in all_features], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    ax.axhline(y=0, color=THEME['text_secondary'], linestyle='--', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)
    
    stats = {
        "n_clusters": len(clusters),
        "features_shown": all_features,
        "cluster_sizes": {c: cluster_stats[c]["size"] for c in clusters}
    }
    
    return buf.getvalue(), "Cluster profile chart showing feature means", stats
