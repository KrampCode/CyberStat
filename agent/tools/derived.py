"""
Tracking for columns the agent derives from the data (PC1..PCn, cluster).

PCA and K-means append columns to the working dataframe so later tools can plot
or group by them. Those columns must not leak into analyses that default to
"all numeric columns" - otherwise a correlation matrix or a second clustering run
silently includes the outputs of the first. The derived column names are stored
in df.attrs, which pandas carries through copy().
"""

import numpy as np
import pandas as pd

DERIVED_ATTR = "cyberstat_derived_columns"


def get_derived_columns(df: pd.DataFrame) -> list[str]:
    """Return derived columns that are still present in the dataframe."""
    return [c for c in df.attrs.get(DERIVED_ATTR, []) if c in df.columns]


def mark_derived(df: pd.DataFrame, columns: list[str]) -> None:
    """Record columns as derived (in place)."""
    existing = set(get_derived_columns(df))
    df.attrs[DERIVED_ATTR] = sorted(existing | set(columns))


def drop_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view of the dataframe with derived columns removed."""
    return df.drop(columns=get_derived_columns(df))


def select_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Select numeric columns for analysis.

    Explicitly requested columns are always honored (so PC1,PC2 can be used
    after PCA). With no columns given, all numeric source columns are returned
    and derived columns are excluded.
    """
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        return df[columns].select_dtypes(include=[np.number])
    return drop_derived(df).select_dtypes(include=[np.number])
