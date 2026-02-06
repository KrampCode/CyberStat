"""Data summary model."""

from pydantic import BaseModel, Field


class DataSummary(BaseModel):
    """
    High-level summary of the loaded dataset.
    
    This populates the "Data Summary" section in the side panel.
    """
    filename: str = Field(description="Name of the uploaded file")
    rows: int = Field(description="Number of rows in the dataset")
    columns: int = Field(description="Number of columns")
    column_names: list[str] = Field(description="List of column names")
    numeric_columns: list[str] = Field(default_factory=list, description="Columns with numeric data")
    categorical_columns: list[str] = Field(default_factory=list, description="Columns with categorical data")
    missing_values: dict[str, int] = Field(default_factory=dict, description="Count of missing values per column")
