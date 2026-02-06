"""Recommendation model."""

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    """
    An actionable recommendation from the agent.
    """
    title: str = Field(description="Brief title")
    description: str = Field(description="What to do and why")
    priority: str = Field(default="medium", description="low, medium, or high")
