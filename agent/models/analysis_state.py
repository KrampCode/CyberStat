"""Analysis state model - the main state container."""

from typing import Optional
from pydantic import BaseModel, Field

from .activity_log import ActivityLogEntry
from .data_summary import DataSummary
from .finding import Finding
from .recommendation import Recommendation


class AnalysisState(BaseModel):
    """
    The complete state of an analysis session.
    
    Tracks data summary, findings, recommendations, and activity log.
    Stored in Streamlit's session state and rendered by the side panel.
    """
    # Data info (populated when user uploads a file)
    data_summary: Optional[DataSummary] = None
    
    # Analysis results (populated as agent discovers insights)
    findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    
    # Activity tracking
    activity_log: list[ActivityLogEntry] = Field(default_factory=list)
    
    def add_activity(self, action: str, details: Optional[str] = None) -> None:
        """Helper to add an activity log entry."""
        self.activity_log.append(ActivityLogEntry(action=action, details=details))
    
    def add_finding(self, title: str, description: str, importance: str = "medium", 
                    related_columns: Optional[list[str]] = None) -> None:
        """Helper to add a finding."""
        self.findings.append(Finding(
            title=title,
            description=description,
            importance=importance,
            related_columns=related_columns or []
        ))
    
    def add_recommendation(self, title: str, description: str, priority: str = "medium") -> None:
        """Helper to add a recommendation."""
        self.recommendations.append(Recommendation(
            title=title,
            description=description,
            priority=priority
        ))
