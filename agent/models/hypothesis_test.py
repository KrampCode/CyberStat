"""Hypothesis test record model."""

from datetime import datetime
from pydantic import BaseModel, Field


class HypothesisTest(BaseModel):
    """
    A hypothesis test run during the session.

    Every p-value the agent sees is recorded so the session can apply a
    multiple-testing correction across all of them.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    test: str = Field(description="Name of the test (e.g., \"Welch's t-test\")")
    hypothesis: str = Field(description="What was tested (e.g., 'salary differs by gender')")
    p_value: float = Field(description="Unadjusted p-value")
