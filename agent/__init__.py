"""
Agent package - The PydanticAI data analytics agent.

Main exports:
- analyst_agent: The configured PydanticAI agent
- run_analyst: Helper function to run the agent
- AnalysisDependencies: Dependencies dataclass for tools
"""

from .analyst import analyst_agent, run_analyst, AnalysisDependencies

__all__ = [
    "analyst_agent",
    "run_analyst",
    "AnalysisDependencies",
]
