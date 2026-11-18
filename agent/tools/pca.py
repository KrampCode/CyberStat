"""PCA (Principal Component Analysis) tool."""

import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .derived import get_derived_columns, mark_derived, select_numeric


def perform_pca(
    df: pd.DataFrame, 
    columns: list[str] = None,
    n_components: int = None
) -> tuple[pd.DataFrame, dict]:
    """
    Perform PCA on numeric columns.
    
    Args:
        df: The dataframe
        columns: List of columns to include (None = all numeric)
        n_components: Number of components to keep (None = all)
        
    Returns:
        Tuple of (transformed_df with PC columns, statistics_dict)
    """
    numeric_df = select_numeric(df, columns)
    
    if len(numeric_df.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for PCA")
    
    # Drop rows with missing values
    clean_df = numeric_df.dropna()
    if len(clean_df) < 10:
        raise ValueError("Need at least 10 complete rows for PCA")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df)
    
    # Determine number of components
    max_components = min(len(clean_df.columns), len(clean_df))
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled_data)
    
    # Create result dataframe with PC columns
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    
    # Drop PC columns from any previous run so stale components don't linger
    stale_pcs = [c for c in get_derived_columns(df) if re.fullmatch(r"PC\d+", c)]
    result_df = df.drop(columns=stale_pcs)
    
    # Add PC columns (NaN for rows that were dropped)
    for i, col in enumerate(pc_columns):
        result_df[col] = np.nan
        result_df.loc[clean_df.index, col] = transformed[:, i]
    mark_derived(result_df, pc_columns)
    
    # Build statistics
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Feature loadings (correlation of features with components)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pc_columns,
        index=clean_df.columns
    )
    
    # Find top contributing features for each component
    top_features = {}
    for i, pc in enumerate(pc_columns[:3]):  # Top 3 components
        abs_loadings = abs(loadings[pc]).sort_values(ascending=False)
        top_features[pc] = [
            {"feature": feat, "loading": round(loadings.loc[feat, pc], 3)}
            for feat in abs_loadings.head(3).index
        ]
    
    stats = {
        "n_samples": len(clean_df),
        "n_features": len(clean_df.columns),
        "n_components": n_components,
        "features_used": list(clean_df.columns),
        "explained_variance": {
            pc_columns[i]: {
                "variance": round(float(explained_var[i]), 4),
                "percentage": round(float(explained_var[i] * 100), 1),
                "cumulative": round(float(cumulative_var[i] * 100), 1)
            }
            for i in range(min(n_components, 5))  # Report up to 5
        },
        "top_features_by_component": top_features,
        "components_for_90_variance": int(np.argmax(cumulative_var >= 0.90) + 1) if any(cumulative_var >= 0.90) else n_components,
        "components_for_95_variance": int(np.argmax(cumulative_var >= 0.95) + 1) if any(cumulative_var >= 0.95) else n_components,
    }
    
    return result_df, stats
