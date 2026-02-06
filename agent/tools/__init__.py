"""
Tools package - Analysis and visualization tools for the agent.

Re-exports all tool functions for convenient importing:
    from agent.tools import describe_dataframe, create_histogram, etc.
"""

# Statistical analysis tools
from .describe import describe_dataframe
from .column_info import get_column_info
from .correlation import calculate_correlation
from .ttest import perform_ttest
from .normality import check_normality
from .value_counts import get_value_counts
from .group_statistics import get_group_statistics

# Machine learning tools
from .pca import perform_pca
from .kmeans import perform_kmeans, find_optimal_k

# Visualization tools
from .visualizations import (
    create_histogram,
    create_scatter_plot,
    create_correlation_heatmap,
    create_box_plot,
    create_bar_chart,
    # PCA visualizations
    create_pca_scatter,
    create_scree_plot,
    # Clustering visualizations
    create_cluster_scatter,
    create_elbow_plot,
    create_cluster_profile_chart,
)

__all__ = [
    # Statistical
    "describe_dataframe",
    "get_column_info",
    "calculate_correlation",
    "perform_ttest",
    "check_normality",
    "get_value_counts",
    "get_group_statistics",
    # Machine learning
    "perform_pca",
    "perform_kmeans",
    "find_optimal_k",
    # Visualizations
    "create_histogram",
    "create_scatter_plot",
    "create_correlation_heatmap",
    "create_box_plot",
    "create_bar_chart",
    "create_pca_scatter",
    "create_scree_plot",
    "create_cluster_scatter",
    "create_elbow_plot",
    "create_cluster_profile_chart",
]
