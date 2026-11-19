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

# Inferential tools (return InferenceResult with p-values for the session registry)
from .group_comparison import compare_groups
from .independence import independence_test
from .regression import fit_ols
from .multiple_testing import InferenceResult, PValue, benjamini_hochberg, summarize_session_tests

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
    # Regression diagnostics
    create_residual_plots,
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
    # Inferential
    "compare_groups",
    "independence_test",
    "fit_ols",
    "InferenceResult",
    "PValue",
    "benjamini_hochberg",
    "summarize_session_tests",
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
    "create_residual_plots",
]
