"""Column info tool."""

import pandas as pd


def get_column_info(df: pd.DataFrame, column_name: str) -> str:
    """
    Get detailed information about a specific column.
    
    Args:
        df: The dataframe
        column_name: Name of the column to analyze
        
    Returns:
        Detailed stats about the column
    """
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found. Available columns: {list(df.columns)}"
    
    col = df[column_name]
    info_parts = [f"Column: {column_name}", f"Data type: {col.dtype}"]
    
    # Basic counts
    info_parts.append(f"Total values: {len(col)}")
    info_parts.append(f"Missing values: {col.isna().sum()} ({col.isna().mean()*100:.1f}%)")
    info_parts.append(f"Unique values: {col.nunique()}")
    
    # Type-specific stats
    if pd.api.types.is_numeric_dtype(col):
        info_parts.extend([
            f"Mean: {col.mean():.2f}",
            f"Std: {col.std():.2f}",
            f"Min: {col.min():.2f}",
            f"25%: {col.quantile(0.25):.2f}",
            f"50% (median): {col.median():.2f}",
            f"75%: {col.quantile(0.75):.2f}",
            f"Max: {col.max():.2f}",
        ])
        
        # Check for outliers using IQR
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((col < q1 - 1.5*iqr) | (col > q3 + 1.5*iqr)).sum()
        info_parts.append(f"Potential outliers (IQR method): {outliers}")
        
    elif pd.api.types.is_categorical_dtype(col) or col.dtype == 'object':
        # Show value counts for categorical
        value_counts = col.value_counts().head(10)
        info_parts.append(f"Top values:\n{value_counts.to_string()}")
    
    return "\n".join(info_parts)
