"""Finding model."""

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """
    A single analytical finding or insight.
    
    Findings are discovered by the agent during analysis and
    populate the "Key Findings" section.
    """
    title: str = Field(description="Brief title of the finding")
    description: str = Field(description="Detailed explanation")
    importance: str = Field(default="medium", description="low, medium, or high")
    related_columns: list[str] = Field(default_factory=list, description="Columns this finding relates to")
