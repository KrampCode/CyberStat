"""Activity log entry model."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ActivityLogEntry(BaseModel):
    """
    A single entry in the activity log.
    
    The activity log shows what the agent is doing in real-time,
    giving users visibility into the analysis process.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str = Field(description="What the agent did (e.g., 'Calculated correlation matrix')")
    details: Optional[str] = Field(default=None, description="Additional context or results snippet")
