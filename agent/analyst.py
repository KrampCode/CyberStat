"""
The Data Analytics Agent - powered by PydanticAI and Gemini.

The agent acts as a statistical analyst that can:
- Understand uploaded datasets
- Run statistical analyses via tools
- Provide insights and recommendations
"""

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from .models import AnalysisState
from . import tools

load_dotenv()


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

## Machine Learning Tools (PCA & Clustering)
Use these for deeper pattern discovery and customer segmentation.

**ALWAYS visualize ML results** - users need to SEE the analysis, not just read numbers!

**PCA (Principal Component Analysis):**
- `run_pca()` - Reduce dimensionality, find which features drive variance
- `create_pca_plot()` - **ALWAYS call this after run_pca()** to show the 2D projection

**K-Means Clustering:**
- `find_optimal_clusters()` - Elbow method (automatically creates elbow plot)
- `run_kmeans()` - Segment data into groups
- `create_cluster_plot()` - **ALWAYS call this after run_kmeans()** to show cluster assignments

**REQUIRED Pipeline (follow ALL steps):**
1. `run_pca()` on relevant numeric features
2. `create_pca_plot()` to show the projection -- DON'T SKIP
3. `find_optimal_clusters()` using PC1,PC2 (creates elbow plot automatically)
4. `run_kmeans()` with the suggested k, using PC1,PC2
5. `create_cluster_plot()` with PC1, PC2 -- DON'T SKIP
6. `record_finding()` with cluster interpretations

The user should see: PCA scatter -> Elbow plot -> Cluster scatter. Three visualizations minimum!

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
    model="google-gla:gemini-3-pro-preview",
    system_prompt=SYSTEM_PROMPT,
    deps_type=AnalysisDependencies,
)


# =============================================================================
# Tool Registration
# =============================================================================


@analyst_agent.tool
def describe_data(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    Get a comprehensive statistical summary of the loaded dataset.
    Use this to understand the overall structure and distributions in the data.
    """
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    # Log the activity
    if ctx.deps.state:
        ctx.deps.state.add_activity("Generated data summary", "Running describe() on all columns")
    
    return tools.describe_dataframe(ctx.deps.df)




@analyst_agent.tool
def analyze_column(ctx: RunContext[AnalysisDependencies], column_name: str) -> str:
    """
    Get detailed statistics and information about a specific column.
    
    Args:
        column_name: The name of the column to analyze
    """
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    if ctx.deps.state:
        ctx.deps.state.add_activity(f"Analyzed column: {column_name}")
    
    return tools.get_column_info(ctx.deps.df, column_name)


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    if ctx.deps.state:
        ctx.deps.state.add_activity(f"Calculated {method} correlations")
    
    return tools.calculate_correlation(ctx.deps.df, method)


@analyst_agent.tool
def compare_groups(
    ctx: RunContext[AnalysisDependencies],
    numeric_column: str,
    group_column: str
) -> str:
    """
    Compare a numeric variable across two groups using a t-test.
    Useful for answering questions like "Is there a significant difference in X between groups A and B?"
    
    Args:
        numeric_column: The numeric column to compare (e.g., 'salary', 'score')
        group_column: The categorical column that defines the groups (must have exactly 2 unique values)
    """
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    if ctx.deps.state:
        ctx.deps.state.add_activity(
            f"Compared {numeric_column} across {group_column}",
            "Running independent samples t-test"
        )
    
    return tools.perform_ttest(ctx.deps.df, numeric_column, group_column)


@analyst_agent.tool
def test_normality(ctx: RunContext[AnalysisDependencies], column_name: str) -> str:
    """
    Test whether a numeric column follows a normal distribution.
    Important for deciding which statistical tests are appropriate.
    
    Args:
        column_name: The numeric column to test
    """
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    if ctx.deps.state:
        ctx.deps.state.add_activity(f"Tested normality of {column_name}")
    
    return tools.check_normality(ctx.deps.df, column_name)


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    if ctx.deps.state:
        ctx.deps.state.add_activity(f"Generated frequency table for {column_name}")
    
    return tools.get_value_counts(ctx.deps.df, column_name, top_n)


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    if ctx.deps.state:
        ctx.deps.state.add_activity(
            f"Calculated group statistics: {numeric_column} by {group_column}"
        )
    
    return tools.get_group_statistics(ctx.deps.df, numeric_column, group_column)


@analyst_agent.tool
def list_columns(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    List all columns in the dataset with their data types.
    Use this to see what variables are available for analysis.
    """
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."
    
    df = ctx.deps.df
    lines = ["Available columns:\n"]
    
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        lines.append(f"  - {col} ({dtype}){f' - {null_pct:.1f}% missing' if null_count > 0 else ''}")
    
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

@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        image_bytes, description, stats = tools.create_histogram(ctx.deps.df, column, bins)
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(f"Created histogram for {column}")
        
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
    except Exception as e:
        return f"Error creating histogram: {e}"


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        color = color_column if color_column else None
        image_bytes, description, stats = tools.create_scatter_plot(ctx.deps.df, x_column, y_column, color)
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(f"Created scatter plot: {y_column} vs {x_column}")
        
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
    except Exception as e:
        return f"Error creating scatter plot: {e}"


@analyst_agent.tool
def create_correlation_heatmap(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    Create a correlation heatmap showing relationships between all numeric columns.
    Returns all correlations ranked by strength - USE THESE for accurate descriptions.
    """
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        image_bytes, description, stats = tools.create_correlation_heatmap(ctx.deps.df)
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity("Created correlation heatmap")
        
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
    except Exception as e:
        return f"Error creating heatmap: {e}"


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        group = group_column if group_column else None
        image_bytes, description, stats = tools.create_box_plot(ctx.deps.df, column, group)
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            activity = f"Created box plot for {column}"
            if group:
                activity += f" by {group}"
            ctx.deps.state.add_activity(activity)
        
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
    except Exception as e:
        return f"Error creating box plot: {e}"


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        image_bytes, description, stats = tools.create_bar_chart(ctx.deps.df, column, top_n)
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(f"Created bar chart for {column}")
        
        # Build response with statistics
        result = f"Bar chart created for {column} - visible in Visualizations panel.\n\n"
        result += f"CATEGORY STATISTICS - Use these in your response:\n"
        result += f"Total records: {stats['total_records']:,} | Unique values: {stats['unique_values']}\n\n"
        
        result += "VALUE COUNTS:\n"
        for val, data in stats["value_counts"].items():
            result += f"  - {val}: {data['count']:,} ({data['percentage']}%)\n"
        
        return result
    except Exception as e:
        return f"Error creating bar chart: {e}"


# =============================================================================
# Machine Learning Tools
# =============================================================================

@analyst_agent.tool
def run_pca(
    ctx: RunContext[AnalysisDependencies],
    columns: str = "",
    n_components: int = 0
) -> str:
    """
    Perform Principal Component Analysis (PCA) to reduce dimensionality and find patterns.
    Use this when you want to understand which features drive the most variance in the data,
    or as a preprocessing step before clustering.
    
    Args:
        columns: Comma-separated list of columns to include (empty = all numeric)
        n_components: Number of components to keep (0 = all)
    """
    if ctx.deps.df is None:
        return "No data loaded."
    
    try:
        # Parse columns
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        n_comp = n_components if n_components > 0 else None
        
        result_df, stats = tools.perform_pca(ctx.deps.df, col_list, n_comp)
        
        # Store the transformed dataframe for subsequent operations
        ctx.deps.df = result_df
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(
                f"Performed PCA",
                f"{stats['n_components']} components from {stats['n_features']} features"
            )
        
        # Build response with statistics
        result = f"PCA COMPLETE - {stats['n_components']} components extracted from {stats['n_features']} features\n"
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
        
        result += "\nPC columns (PC1, PC2, etc.) have been added to the dataframe."
        result += "\n\nNEXT STEP: Call create_pca_plot() to visualize the projection!"
        
        return result
    except Exception as e:
        return f"Error performing PCA: {e}"


@analyst_agent.tool
def run_kmeans(
    ctx: RunContext[AnalysisDependencies],
    columns: str = "",
    n_clusters: int = 3
) -> str:
    """
    Perform K-means clustering to segment data into groups.
    Use this to find natural groupings in the data (e.g., customer segments).
    TIP: Run PCA first if you have many features, then cluster on PC1 and PC2.
    
    Args:
        columns: Comma-separated list of columns to cluster on (empty = all numeric, or use PC1,PC2 after PCA)
        n_clusters: Number of clusters (default 3, use find_optimal_clusters to determine)
    """
    if ctx.deps.df is None:
        return "No data loaded."
    
    try:
        # Parse columns
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        
        result_df, stats = tools.perform_kmeans(ctx.deps.df, col_list, n_clusters)
        
        # Store the clustered dataframe
        ctx.deps.df = result_df
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(
                f"K-means clustering",
                f"{n_clusters} clusters from {stats['n_samples']:,} samples"
            )
        
        # Build response with statistics
        result = f"K-MEANS CLUSTERING COMPLETE\n"
        result += f"Clusters: {stats['n_clusters']} | Samples: {stats['n_samples']:,}\n"
        result += f"Features used: {', '.join(stats['features_used'])}\n"
        result += f"Inertia: {stats['inertia']}\n\n"
        
        result += "CLUSTER SIZES:\n"
        for cluster, info in stats["cluster_stats"].items():
            result += f"  - {cluster}: {info['size']:,} samples ({info['percentage']}%)\n"
        
        result += "\nDISTINGUISHING FEATURES BY CLUSTER:\n"
        for cluster, features in stats["distinguishing_features"].items():
            result += f"  {cluster}:\n"
            for f in features:
                result += f"    - {f['feature']}: {f['direction']} (cluster mean: {f['cluster_mean']}, global: {f['global_mean']})\n"
        
        result += "\nA 'cluster' column has been added to the dataframe."
        result += "\n\nNEXT STEP: Call create_cluster_plot(x_column='PC1', y_column='PC2') to visualize!"
        
        return result
    except Exception as e:
        return f"Error performing K-means: {e}"


@analyst_agent.tool
def find_optimal_clusters(
    ctx: RunContext[AnalysisDependencies],
    columns: str = "",
    max_k: int = 10
) -> str:
    """
    Use the elbow method to find the optimal number of clusters for K-means.
    Run this BEFORE run_kmeans to determine how many clusters to use.
    
    Args:
        columns: Comma-separated list of columns (empty = all numeric)
        max_k: Maximum number of clusters to try (default 10)
    """
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        
        stats = tools.find_optimal_k(ctx.deps.df, col_list, max_k)
        
        # Create elbow plot
        image_bytes, description, plot_stats = tools.create_elbow_plot(
            stats["k_values"], 
            stats["inertias"],
            stats["suggested_k"]
        )
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(f"Elbow analysis", f"Suggested k={stats['suggested_k']}")
        
        result = f"ELBOW ANALYSIS COMPLETE - Visible in Visualizations panel\n\n"
        result += f"Samples: {stats['n_samples']:,}\n"
        result += f"Features: {', '.join(stats['features_used'])}\n\n"
        result += f"SUGGESTED OPTIMAL K: {stats['suggested_k']}\n\n"
        result += "INERTIA BY K:\n"
        for k, inertia in zip(stats["k_values"], stats["inertias"]):
            marker = " <-- suggested" if k == stats["suggested_k"] else ""
            result += f"  k={k}: {inertia:,.0f}{marker}\n"
        
        return result
    except Exception as e:
        return f"Error finding optimal clusters: {e}"


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        color = color_by if color_by else None
        image_bytes, description, stats = tools.create_pca_scatter(
            ctx.deps.df, 'PC1', 'PC2', color
        )
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(f"Created PCA scatter plot")
        
        result = f"PCA scatter plot created - visible in Visualizations panel.\n\n"
        result += f"DATA POINTS: {stats['n_points']:,}\n"
        result += f"PC1 range: {stats['pc1_range'][0]} to {stats['pc1_range'][1]}\n"
        result += f"PC2 range: {stats['pc2_range'][0]} to {stats['pc2_range'][1]}\n"
        
        if "colored_by" in stats:
            result += f"Colored by: {stats['colored_by']} ({stats['n_categories']} categories)"
        
        return result
    except Exception as e:
        return f"Error creating PCA plot: {e}"


@analyst_agent.tool
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
    if ctx.deps.df is None:
        return "No data loaded."
    if ctx.deps.plots is None:
        return "Visualization storage not available."
    
    try:
        image_bytes, description, stats = tools.create_cluster_scatter(
            ctx.deps.df, x_column, y_column, 'cluster'
        )
        ctx.deps.plots.append((image_bytes, description))
        
        if ctx.deps.state:
            ctx.deps.state.add_activity(f"Created cluster scatter plot")
        
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
    except Exception as e:
        return f"Error creating cluster plot: {e}"


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
