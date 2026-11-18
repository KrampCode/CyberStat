"""Describe dataframe tool."""

import pandas as pd

from .derived import drop_derived


def describe_dataframe(df: pd.DataFrame) -> str:
    """
    Get a statistical summary of the dataframe.
    
    Returns pandas describe() output formatted as a string,
    plus additional info about data types. Columns derived by PCA/clustering
    are excluded so the summary describes the source data.
    """
    df = drop_derived(df)
    
    # Basic describe for numeric columns
    desc = df.describe(include='all').round(2).to_string()
    
    # Add data type info
    dtypes = df.dtypes.to_string()
    
    return f"Statistical Summary:\n{desc}\n\nData Types:\n{dtypes}"
