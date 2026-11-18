"""
The Data Analytics Agent - powered by PydanticAI and Gemini.

The agent acts as a statistical analyst that can:
- Understand uploaded datasets
- Run statistical analyses via tools
- Provide insights and recommendations
"""

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from .models import AnalysisState
from . import tools
from .tools.derived import get_derived_columns

load_dotenv()

# Any PydanticAI model string works, e.g. "anthropic:claude-sonnet-5-5" or "openai:gpt-5"
DEFAULT_MODEL = "google:gemini-3-pro-preview"
MODEL = os.getenv("CYBERSTAT_MODEL", DEFAULT_MODEL)

NO_DATA_MESSAGE = "No data loaded. Please ask the user to upload a dataset first."


@dataclass
class AnalysisDependencies:
    """
    Dependencies passed to the agent and its tools via RunContext.
    
    Provides access to:
    - The current dataframe being analyzed
    - The analysis state (to log activities, add findings)
    - The plots list (to store generated visualizations)
    """
    df: Optional[pd.DataFrame] = None
    state: Optional[AnalysisState] = None
    plots: Optional[list] = None  # List of (image_bytes, description) tuples


SYSTEM_PROMPT = """You are a friendly and insightful statistical analyst assistant. 
Your role is to help users understand their data through conversation.

## Your Capabilities
- Analyze datasets uploaded by the user
- Explain statistical concepts in accessible terms
- Identify patterns, trends, and anomalies
- Provide actionable recommendations based on data
- Create visualizations to support your findings

## How to Behave
1. When a user uploads data, start by understanding what they want to learn
2. Use your analysis tools to examine the data - don't guess at statistics
3. Explain findings in plain language, then provide technical details
4. Proactively suggest relevant analyses based on the data type
5. When you discover interesting findings, highlight them clearly

## CRITICAL: DATA VERIFICATION RULES
**NEVER make claims about data without first getting statistics from tools.**

Before stating anything about the data, you MUST:
1. **For group comparisons**: ALWAYS use `get_group_stats()` or `compare_groups()` FIRST
   - Never say groups are "similar" or "different" without actual numbers
   - Always cite the specific means/medians when comparing
   
2. **For correlations**: ALWAYS use `get_correlations()` or check scatter plot stats
   - Never guess at correlation strength - use the actual r value
   
3. **For distributions**: ALWAYS use `describe_data()` or histogram stats
   - Never describe shape without actual skewness/mean/median values

**Visualization tools return statistics** - READ THEM before making claims:
- Scatter plots return: correlation r, p-value, slope, interpretation
- Box plots return: per-group mean, median, std, min, max, IQR
- Histograms return: mean, median, std, skewness, interpretation
- Bar charts return: counts and percentages per category

WRONG: "The groups appear to have similar values" (guessing)
RIGHT: "Bronze members average 4.2 purchases vs Platinum's 32.1 - a 7.6x difference" (from stats)

## Recording Insights & Visualizations
When you discover something noteworthy during analysis, you MUST use:
- `record_finding()` - to save key insights (correlations, patterns, anomalies, etc.)
- `record_recommendation()` - to suggest next steps or actions

**Create visualizations** to back up your findings using:
- `create_histogram()` - for distributions of numeric columns
- `create_scatter()` - for relationships between two variables
- `create_correlation_heatmap()` - for overview of all correlations
- `create_boxplot()` - for comparing distributions or showing outliers
- `create_bar_chart()` - for categorical data frequencies

Visualizations appear in a dedicated panel for the user. Generate them proactively!
For example:
- Found a strong correlation? Create a scatter plot AND record the finding.
- Analyzing a distribution? Create a histogram.
- Comparing groups? Create a box plot AND cite the statistics it returns.

## Statistical Testing
- `compare_groups()` runs Welch's t-test (no equal-variance assumption) and reports the
  mean difference with a 95% confidence interval and Cohen's d / Hedges' g.
- Always report the effect size and CI, not just the p-value. A significant p-value with a
  negligible effect size is not a practically important difference.
- Use `test_normality()` to inform test choice, but note that with large samples the
  Shapiro-Wilk test flags trivial departures - judge by skewness/kurtosis too.

## Machine Learning Tools (PCA & Clustering)
Use these when the user asks about segments, groupings, structure, or dimensionality -
not by default for every question.

**PCA (Principal Component Analysis):**
- `run_pca()` - find which features drive variance; adds PC1..PCn columns
- `create_pca_plot()` - show the 2D projection (call after `run_pca()`)

**K-Means Clustering:**
- `find_optimal_clusters()` - compares k by silhouette score (primary) and elbow (secondary);
  creates an elbow plot automatically
- `run_kmeans()` - segment the data; adds a `cluster` column and reports the silhouette score
- `create_cluster_plot()` or `create_pca_plot(color_by='cluster')` - visualize the segments

**Recommended segmentation workflow:**
1. Choose the features relevant to the question (avoid IDs and derived columns)
2. `find_optimal_clusters()` on those features
3. `run_kmeans()` on the same features with the suggested k
4. `run_pca()` on the same features, then `create_pca_plot(color_by='cluster')` to visualize
5. `record_finding()` with cluster profiles, citing the silhouette score

Cluster on the standardized original features, not just PC1/PC2 - two components may
discard much of the variance. Only cluster on PCs if many features are highly correlated,
and then use enough components to explain ~90% of variance.
If the best silhouette is below 0.25, tell the user the data has no strong cluster
structure rather than presenting the segments as real.

PC and cluster columns are derived: tools that default to "all numeric columns" exclude
them automatically. Pass them explicitly (e.g. columns="PC1,PC2") when you need them.

## Communication Style
- Be conversational but professional
- Use bullet points for clarity when listing multiple items
- Include specific numbers from your analysis (ALWAYS cite actual values)
- Acknowledge limitations in the data or analysis

## Important
- Always use tools to get actual statistics - never make up numbers
- If the user hasn't uploaded data yet, guide them to do so
- If asked about something outside the data, be honest about limitations
"""


# Create the agent with Gemini
analyst_agent = Agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    deps_type=AnalysisDependencies,
    # Resolve the model on first run, so the module imports without an API key
    defer_model_check=True,
)


def log_activity(ctx: RunContext[AnalysisDependencies], action: str, details: Optional[str] = None) -> None:
    """Record an action in the activity log, if analysis state is available."""
    if ctx.deps.state:
        ctx.deps.state.add_activity(action, details)


def data_tool(needs_plots: bool = False) -> Callable:
    """
    Register an agent tool that operates on the loaded dataframe.
    
    Handles the checks every analysis tool needs: that data is loaded, that plot
    storage exists (for visualization tools), and that errors are returned to the
    model as text it can recover from rather than crashing the run.
    """
    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(func)
        def wrapper(ctx: RunContext[AnalysisDependencies], *args, **kwargs) -> str:
            if ctx.deps.df is None:
                return NO_DATA_MESSAGE
            if needs_plots and ctx.deps.plots is None:
                return "Visualization storage not available."
            try:
                return func(ctx, *args, **kwargs)
            except Exception as e:
                return f"Error in {func.__name__}: {e}"
        return analyst_agent.tool(wrapper)
    return decorator


# =============================================================================
# Tool Registration
# =============================================================================


@data_tool()
def describe_data(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    Get a comprehensive statistical summary of the loaded dataset.
    Use this to understand the overall structure and distributions in the data.
    """
    log_activity(ctx, "Generated data summary", "Running describe() on all columns")
    return tools.describe_dataframe(ctx.deps.df)


@data_tool()
def analyze_column(ctx: RunContext[AnalysisDependencies], column_name: str) -> str:
    """
    Get detailed statistics and information about a specific column.

    Args:
        column_name: The name of the column to analyze
    """
    log_activity(ctx, f"Analyzed column: {column_name}")
    return tools.get_column_info(ctx.deps.df, column_name)


@data_tool()
def get_correlations(
    ctx: RunContext[AnalysisDependencies],
    method: str = "pearson"
) -> str:
    """
    Calculate correlations between all numeric columns in the dataset.

    Args:
        method: The correlation method to use - 'pearson' (linear relationships),
                'spearman' (monotonic relationships), or 'kendall' (ordinal data)
    """
    log_activity(ctx, f"Calculated {method} correlations")
    return tools.calculate_correlation(ctx.deps.df, method)


@data_tool()
def compare_groups(
    ctx: RunContext[AnalysisDependencies],
    numeric_column: str,
    group_column: str
) -> str:
    """
    Compare a numeric variable across two groups using Welch's t-test.
    Useful for answering questions like "Is there a significant difference in X between groups A and B?"
    Returns the mean difference with a 95% confidence interval and effect size (Cohen's d, Hedges' g).

    Args:
        numeric_column: The numeric column to compare (e.g., 'salary', 'score')
        group_column: The categorical column that defines the groups (must have exactly 2 unique values)
    """
    log_activity(ctx, f"Compared {numeric_column} across {group_column}", "Running Welch's t-test")
    return tools.perform_ttest(ctx.deps.df, numeric_column, group_column)


@data_tool()
def test_normality(ctx: RunContext[AnalysisDependencies], column_name: str) -> str:
    """
    Test whether a numeric column follows a normal distribution.
    Important for deciding which statistical tests are appropriate.

    Args:
        column_name: The numeric column to test
    """
    log_activity(ctx, f"Tested normality of {column_name}")
    return tools.check_normality(ctx.deps.df, column_name)


@data_tool()
def get_frequencies(
    ctx: RunContext[AnalysisDependencies],
    column_name: str,
    top_n: int = 20
) -> str:
    """
    Get the frequency distribution of values in a column.
    Shows how often each value appears, useful for understanding categorical data.

    Args:
        column_name: The column to analyze
        top_n: Number of most frequent values to show (default 20)
    """
    log_activity(ctx, f"Generated frequency table for {column_name}")
    return tools.get_value_counts(ctx.deps.df, column_name, top_n)


@data_tool()
def get_group_stats(
    ctx: RunContext[AnalysisDependencies],
    numeric_column: str,
    group_column: str
) -> str:
    """
    Get detailed statistics for a numeric column grouped by a categorical column.
    ESSENTIAL for comparing groups - use this BEFORE making any claims about group differences.
    Returns mean, median, std, min, max, IQR for each group plus a comparison summary.

    Args:
        numeric_column: The numeric column to analyze (e.g., 'salary', 'purchases')
        group_column: The categorical column to group by (e.g., 'department', 'tier')
    """
    log_activity(ctx, f"Calculated group statistics: {numeric_column} by {group_column}")
    return tools.get_group_statistics(ctx.deps.df, numeric_column, group_column)


@data_tool()
def list_columns(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    List all columns in the dataset with their data types.
    Use this to see what variables are available for analysis.
    """
    df = ctx.deps.df
    derived = set(get_derived_columns(df))
    lines = ["Available columns:\n"]

    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        missing = f" - {null_pct:.1f}% missing" if null_count > 0 else ""
        source = " [derived by PCA/clustering]" if col in derived else ""
        lines.append(f"  - {col} ({dtype}){missing}{source}")

    lines.append(f"\nTotal: {len(df)} rows, {len(df.columns)} columns")

    return "\n".join(lines)


# =============================================================================
# Tools for Recording Insights
# =============================================================================

@analyst_agent.tool
def record_finding(
    ctx: RunContext[AnalysisDependencies],
    title: str,
    description: str,
    importance: str = "medium",
    related_columns: str = ""
) -> str:
    """
    Record a key finding or insight discovered during analysis.
    Use this whenever you discover something noteworthy about the data.
    The finding will be displayed in the Analysis panel for the user.

    Args:
        title: Brief title for the finding (e.g., "Strong correlation detected")
        description: Detailed explanation of what was found and why it matters
        importance: How important is this finding - "low", "medium", or "high"
        related_columns: Comma-separated list of column names this finding relates to
    """
    if ctx.deps.state is None:
        return "Unable to record finding - no analysis state available."

    # Parse related columns from comma-separated string
    columns = [c.strip() for c in related_columns.split(",") if c.strip()] if related_columns else []

    ctx.deps.state.add_finding(
        title=title,
        description=description,
        importance=importance,
        related_columns=columns
    )

    ctx.deps.state.add_activity(f"Recorded finding: {title}")

    return f"Finding recorded: {title}"


@analyst_agent.tool
def record_recommendation(
    ctx: RunContext[AnalysisDependencies],
    title: str,
    description: str,
    priority: str = "medium"
) -> str:
    """
    Record a recommendation for the user based on the analysis.
    Use this to suggest next steps, further analyses, or actions to take.
    The recommendation will be displayed in the Analysis panel.

    Args:
        title: Brief title for the recommendation (e.g., "Investigate outliers")
        description: What the user should do and why
        priority: How urgent is this - "low", "medium", or "high"
    """
    if ctx.deps.state is None:
        return "Unable to record recommendation - no analysis state available."

    ctx.deps.state.add_recommendation(
        title=title,
        description=description,
        priority=priority
    )

    ctx.deps.state.add_activity(f"Added recommendation: {title}")

    return f"Recommendation recorded: {title}"


# =============================================================================
# Visualization Tools
# =============================================================================

@data_tool(needs_plots=True)
def create_histogram(
    ctx: RunContext[AnalysisDependencies],
    column: str,
    bins: int = 30
) -> str:
    """
    Create a histogram to visualize the distribution of a numeric column.
    Returns statistics including mean, median, std, skewness - USE THESE for accurate descriptions.

    Args:
        column: The numeric column to plot
        bins: Number of bins (default 30)
    """
    image_bytes, description, stats = tools.create_histogram(ctx.deps.df, column, bins)
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, f"Created histogram for {column}")

    # Return statistics for the agent to use in its response
    return f"""Histogram created for {column} (visible in Visualizations panel).

DISTRIBUTION STATISTICS - Use these in your response:
- Count: {stats['count']:,}
- Mean: {stats['mean']}
- Median: {stats['median']}
- Std Dev: {stats['std']}
- Min: {stats['min']} | Max: {stats['max']}
- Q1: {stats['q1']} | Q3: {stats['q3']} | IQR: {stats['iqr']}
- Skewness: {stats['skewness']} ({stats['skew_interpretation']})"""


@data_tool(needs_plots=True)
def create_scatter(
    ctx: RunContext[AnalysisDependencies],
    x_column: str,
    y_column: str,
    color_column: str = ""
) -> str:
    """
    Create a scatter plot to visualize the relationship between two numeric columns.
    Returns correlation statistics - USE THESE to accurately describe the relationship.

    Args:
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column to color points by (leave empty for no coloring)
    """
    color = color_column if color_column else None
    image_bytes, description, stats = tools.create_scatter_plot(ctx.deps.df, x_column, y_column, color)
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, f"Created scatter plot: {y_column} vs {x_column}")

    # Build response with correlation statistics
    result = f"Scatter plot created ({y_column} vs {x_column}) - visible in Visualizations panel.\n\n"
    result += f"DATA POINTS: {stats['n_points']:,}\n\n"

    if "correlation" in stats:
        corr = stats["correlation"]
        reg = stats["regression"]
        result += f"""CORRELATION STATISTICS - Use these in your response:
- Correlation (r): {corr['r']} ({corr['interpretation']})
- R-squared: {corr['r_squared']} ({corr['r_squared']*100:.1f}% of variance explained)
- P-value: {corr['p_value']} ({'statistically significant' if corr['significant'] else 'not significant'})

REGRESSION:
- {reg['interpretation']}
- Slope: {reg['slope']} | Intercept: {reg['intercept']}"""
    else:
        result += "Insufficient data points for correlation analysis."

    return result


@data_tool(needs_plots=True)
def create_correlation_heatmap(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    Create a correlation heatmap showing relationships between all numeric columns.
    Returns all correlations ranked by strength - USE THESE for accurate descriptions.
    """
    image_bytes, description, stats = tools.create_correlation_heatmap(ctx.deps.df)
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, "Created correlation heatmap")

    # Build response with correlation statistics
    result = f"Correlation heatmap created - visible in Visualizations panel.\n\n"
    result += f"ANALYZED {stats['n_variables']} NUMERIC VARIABLES\n\n"

    summary = stats["summary"]
    result += f"CORRELATION SUMMARY:\n"
    result += f"- Strong (|r| >= 0.7): {summary['strong_count']}\n"
    result += f"- Moderate (0.4 <= |r| < 0.7): {summary['moderate_count']}\n"
    result += f"- Weak (|r| < 0.4): {summary['weak_count']}\n\n"

    if stats["strong_correlations"]:
        result += "STRONG CORRELATIONS - Use these in your response:\n"
        for c in stats["strong_correlations"]:
            result += f"  - {c['pair']}: r = {c['r']}\n"
        result += "\n"

    if stats["moderate_correlations"]:
        result += "MODERATE CORRELATIONS:\n"
        for c in stats["moderate_correlations"][:5]:  # Top 5 moderate
            result += f"  - {c['pair']}: r = {c['r']}\n"

    return result


@data_tool(needs_plots=True)
def create_boxplot(
    ctx: RunContext[AnalysisDependencies],
    column: str,
    group_column: str = ""
) -> str:
    """
    Create a box plot to show the distribution of a numeric column.
    When grouped, returns per-group statistics - USE THESE to accurately compare groups.

    Args:
        column: The numeric column to plot
        group_column: Optional categorical column to group by (leave empty for single box)
    """
    group = group_column if group_column else None
    image_bytes, description, stats = tools.create_box_plot(ctx.deps.df, column, group)
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, f"Created box plot for {column}" + (f" by {group}" if group else ""))

    # Build response with statistics
    result = f"Box plot created - visible in Visualizations panel.\n\n"

    if "groups" in stats:
        result += f"GROUP STATISTICS FOR {column} BY {stats['grouped_by']} - Use these in your response:\n\n"

        # Sort groups by mean for easy comparison
        sorted_groups = sorted(
            stats["groups"].items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )

        for group_name, group_stats in sorted_groups:
            result += f"  {group_name}:\n"
            result += f"   Count: {group_stats['count']:,} | Mean: {group_stats['mean']} | Median: {group_stats['median']}\n"
            result += f"   Std: {group_stats['std']} | Range: {group_stats['min']} - {group_stats['max']}\n"
            result += f"   IQR: {group_stats['q1']} - {group_stats['q3']}\n\n"

        # Add comparison if multiple groups
        if len(sorted_groups) >= 2:
            highest = sorted_groups[0]
            lowest = sorted_groups[-1]
            diff = highest[1]["mean"] - lowest[1]["mean"]
            result += f"COMPARISON: {highest[0]} has highest mean ({highest[1]['mean']}) vs {lowest[0]} ({lowest[1]['mean']})\n"
            result += f"Difference: {diff:.2f}"
    else:
        # Single distribution
        s = stats["overall"]
        result += f"STATISTICS FOR {column}:\n"
        result += f"Count: {s['count']:,} | Mean: {s['mean']} | Median: {s['median']}\n"
        result += f"Std: {s['std']} | Range: {s['min']} - {s['max']}\n"
        result += f"IQR: {s['q1']} - {s['q3']}"

    return result


@data_tool(needs_plots=True)
def create_bar_chart(
    ctx: RunContext[AnalysisDependencies],
    column: str,
    top_n: int = 10
) -> str:
    """
    Create a bar chart showing the frequency of values in a column.
    Returns counts and percentages - USE THESE for accurate descriptions.

    Args:
        column: The column to plot value counts for
        top_n: Number of top values to show (default 10)
    """
    image_bytes, description, stats = tools.create_bar_chart(ctx.deps.df, column, top_n)
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, f"Created bar chart for {column}")

    # Build response with statistics
    result = f"Bar chart created for {column} - visible in Visualizations panel.\n\n"
    result += f"CATEGORY STATISTICS - Use these in your response:\n"
    result += f"Total records: {stats['total_records']:,} | Unique values: {stats['unique_values']}\n\n"

    result += "VALUE COUNTS:\n"
    for val, data in stats["value_counts"].items():
        result += f"  - {val}: {data['count']:,} ({data['percentage']}%)\n"

    return result


# =============================================================================
# Machine Learning Tools
# =============================================================================

def _parse_columns(columns: str) -> Optional[list[str]]:
    """Parse a comma-separated column list; empty means 'all numeric'."""
    return [c.strip() for c in columns.split(",") if c.strip()] or None


@data_tool()
def run_pca(
    ctx: RunContext[AnalysisDependencies],
    columns: str = "",
    n_components: int = 0
) -> str:
    """
    Perform Principal Component Analysis (PCA) to reduce dimensionality and find patterns.
    Use this when you want to understand which features drive the most variance in the data,
    or to visualize high-dimensional data and clusters in 2D.

    Args:
        columns: Comma-separated list of columns to include (empty = all numeric, excluding derived columns)
        n_components: Number of components to keep (0 = all)
    """
    n_comp = n_components if n_components > 0 else None
    result_df, stats = tools.perform_pca(ctx.deps.df, _parse_columns(columns), n_comp)

    # Store the transformed dataframe for subsequent operations
    ctx.deps.df = result_df
    log_activity(ctx, "Performed PCA", f"{stats['n_components']} components from {stats['n_features']} features")

    # Build response with statistics
    result = f"PCA COMPLETE - {stats['n_components']} components extracted from {stats['n_features']} features\n"
    result += f"Features used: {', '.join(stats['features_used'])}\n"
    result += f"Samples analyzed: {stats['n_samples']:,}\n\n"

    result += "EXPLAINED VARIANCE BY COMPONENT:\n"
    for pc, var_info in stats["explained_variance"].items():
        result += f"  - {pc}: {var_info['percentage']}% (cumulative: {var_info['cumulative']}%)\n"

    result += f"\nComponents needed for 90% variance: {stats['components_for_90_variance']}\n"
    result += f"Components needed for 95% variance: {stats['components_for_95_variance']}\n\n"

    result += "TOP CONTRIBUTING FEATURES:\n"
    for pc, features in stats["top_features_by_component"].items():
        result += f"  {pc}:\n"
        for f in features:
            result += f"    - {f['feature']}: {f['loading']}\n"

    result += "\nPC columns (PC1, PC2, etc.) have been added to the dataframe as derived columns."
    result += "\n\nNEXT STEP: Call create_pca_plot() to visualize the projection."

    return result


@data_tool()
def run_kmeans(
    ctx: RunContext[AnalysisDependencies],
    columns: str = "",
    n_clusters: int = 3
) -> str:
    """
    Perform K-means clustering on standardized features to segment data into groups.
    Use this to find natural groupings in the data (e.g., customer segments).
    Reports the silhouette score - use it to judge whether the clusters are real.

    Args:
        columns: Comma-separated list of columns to cluster on (empty = all numeric, excluding derived columns)
        n_clusters: Number of clusters (default 3, use find_optimal_clusters to determine)
    """
    result_df, stats = tools.perform_kmeans(ctx.deps.df, _parse_columns(columns), n_clusters)

    # Store the clustered dataframe
    ctx.deps.df = result_df
    log_activity(ctx, "K-means clustering", f"{n_clusters} clusters from {stats['n_samples']:,} samples")

    # Build response with statistics
    result = f"K-MEANS CLUSTERING COMPLETE\n"
    result += f"Clusters: {stats['n_clusters']} | Samples: {stats['n_samples']:,}\n"
    result += f"Features used: {', '.join(stats['features_used'])}\n"
    result += f"Inertia: {stats['inertia']}\n"
    result += f"Silhouette score: {stats['silhouette']} ({stats['silhouette_interpretation']})\n\n"

    result += "CLUSTER SIZES:\n"
    for cluster, info in stats["cluster_stats"].items():
        result += f"  - {cluster}: {info['size']:,} samples ({info['percentage']}%)\n"

    result += "\nDISTINGUISHING FEATURES BY CLUSTER:\n"
    for cluster, features in stats["distinguishing_features"].items():
        result += f"  {cluster}:\n"
        for f in features:
            result += f"    - {f['feature']}: {f['direction']} (cluster mean: {f['cluster_mean']}, global: {f['global_mean']})\n"

    result += "\nA 'cluster' column has been added to the dataframe as a derived column."
    result += "\n\nNEXT STEP: Visualize with create_pca_plot(color_by='cluster') after run_pca(), or create_cluster_plot()."

    return result


@data_tool(needs_plots=True)
def find_optimal_clusters(
    ctx: RunContext[AnalysisDependencies],
    columns: str = "",
    max_k: int = 10
) -> str:
    """
    Compare candidate cluster counts for K-means using silhouette scores and the elbow method.
    Run this BEFORE run_kmeans to determine how many clusters to use.

    Args:
        columns: Comma-separated list of columns (empty = all numeric, excluding derived columns)
        max_k: Maximum number of clusters to try (default 10)
    """
    stats = tools.find_optimal_k(ctx.deps.df, _parse_columns(columns), max_k)

    # Create elbow plot
    image_bytes, description, plot_stats = tools.create_elbow_plot(
        stats["k_values"],
        stats["inertias"],
        stats["suggested_k"]
    )
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, "Cluster count analysis", f"Suggested k={stats['suggested_k']}")

    result = f"CLUSTER COUNT ANALYSIS COMPLETE - elbow plot visible in Visualizations panel\n\n"
    result += f"Samples: {stats['n_samples']:,}\n"
    result += f"Features: {', '.join(stats['features_used'])}\n\n"
    result += f"SUGGESTED K (highest silhouette): {stats['suggested_k']}\n"
    result += f"Best silhouette: {stats['best_silhouette']} ({stats['silhouette_interpretation']})\n"
    if stats["elbow_k"] is not None:
        agreement = "agrees" if stats["elbow_k"] == stats["suggested_k"] else "disagrees"
        result += f"Elbow method suggests k={stats['elbow_k']} ({agreement} with silhouette)\n"

    result += "\nBY K:\n"
    for k, inertia, sil in zip(stats["k_values"], stats["inertias"], stats["silhouettes"]):
        marker = " <-- suggested" if k == stats["suggested_k"] else ""
        result += f"  k={k}: inertia={inertia:,.0f}, silhouette={sil}{marker}\n"

    return result


@data_tool(needs_plots=True)
def create_pca_plot(
    ctx: RunContext[AnalysisDependencies],
    color_by: str = ""
) -> str:
    """
    Create a scatter plot of the first two principal components.
    Run PCA first to add PC1 and PC2 columns to the data.

    Args:
        color_by: Optional column to color points by (e.g., 'cluster' after clustering)
    """
    color = color_by if color_by else None
    image_bytes, description, stats = tools.create_pca_scatter(ctx.deps.df, 'PC1', 'PC2', color)
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, "Created PCA scatter plot")

    result = f"PCA scatter plot created - visible in Visualizations panel.\n\n"
    result += f"DATA POINTS: {stats['n_points']:,}\n"
    result += f"PC1 range: {stats['pc1_range'][0]} to {stats['pc1_range'][1]}\n"
    result += f"PC2 range: {stats['pc2_range'][0]} to {stats['pc2_range'][1]}\n"

    if "colored_by" in stats:
        result += f"Colored by: {stats['colored_by']} ({stats['n_categories']} categories)"

    return result


@data_tool(needs_plots=True)
def create_cluster_plot(
    ctx: RunContext[AnalysisDependencies],
    x_column: str,
    y_column: str
) -> str:
    """
    Create a scatter plot showing cluster assignments.
    Run K-means first to add the 'cluster' column. Use PC1 and PC2 for best visualization.

    Args:
        x_column: Column for x-axis (e.g., 'PC1')
        y_column: Column for y-axis (e.g., 'PC2')
    """
    image_bytes, description, stats = tools.create_cluster_scatter(
        ctx.deps.df, x_column, y_column, 'cluster'
    )
    ctx.deps.plots.append((image_bytes, description))
    log_activity(ctx, "Created cluster scatter plot")

    result = f"Cluster scatter plot created - visible in Visualizations panel.\n\n"
    result += f"DATA POINTS: {stats['n_points']:,}\n"
    result += f"CLUSTERS: {stats['n_clusters']}\n\n"

    result += "CLUSTER SIZES:\n"
    for cluster, size in stats["cluster_sizes"].items():
        result += f"  - {cluster}: {size:,}\n"

    result += "\nCENTROIDS (in plot coordinates):\n"
    for cluster, centroid in stats["centroids"].items():
        result += f"  - {cluster}: ({centroid['x']}, {centroid['y']})\n"

    return result


# =============================================================================
# Helper function to run the agent
# =============================================================================

async def run_analyst(
    user_message: str,
    df: Optional[pd.DataFrame] = None,
    state: Optional[AnalysisState] = None,
    message_history: Optional[list] = None,
    plots: Optional[list] = None
) -> tuple[str, list, list, Optional[pd.DataFrame]]:
    """
    Run the analyst agent with a user message.
    
    Args:
        user_message: The user's question or request
        df: The pandas DataFrame to analyze (if any)
        state: The AnalysisState to update with findings
        message_history: Previous messages for context
        plots: List to store generated plots (will be mutated)
        
    Returns:
        Tuple of (response_text, updated_message_history, new_plots, updated_df)
        - response_text: The agent's response as a string
        - updated_message_history: Full conversation history to pass to next call
        - new_plots: List of (image_bytes, description) tuples generated during this call
        - updated_df: The dataframe (may have new columns from PCA/clustering)
    """
    # Create a fresh list to collect plots from this run
    new_plots = [] if plots is None else plots
    
    deps = AnalysisDependencies(df=df, state=state, plots=new_plots)
    
    result = await analyst_agent.run(
        user_message,
        deps=deps,
        message_history=message_history
    )
    
    # Return the response, message history, plots, and potentially modified dataframe
    return result.output, result.all_messages(), new_plots, deps.df
