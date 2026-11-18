"""
Agent wiring tests. These run the real agent and tools with scripted models,
so no LLM API calls are made.
"""

import asyncio

import matplotlib
matplotlib.use("Agg")

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from agent.analyst import NO_DATA_MESSAGE, analyst_agent, run_analyst
from agent.models import AnalysisState
from agent.tools.derived import get_derived_columns


def _tool_returns(messages: list[ModelMessage]) -> dict[str, str]:
    return {
        part.tool_name: part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


def scripted_model(calls: list[tuple[str, dict]]) -> FunctionModel:
    """A model that makes the given tool calls in order, then replies with text."""
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        n_responses = sum(isinstance(m, ModelResponse) for m in messages)
        if n_responses < len(calls):
            name, args = calls[n_responses]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        return ModelResponse(parts=[TextPart("done")])
    return FunctionModel(respond)


def test_every_tool_runs_without_crashing(iris):
    """TestModel calls every registered tool with generated arguments."""
    state = AnalysisState()
    with analyst_agent.override(model=TestModel()):
        _, messages, _, _ = asyncio.run(run_analyst("analyze", df=iris, state=state, plots=[]))
    returns = _tool_returns(messages)
    assert set(returns) == set(analyst_agent._function_toolset.tools)
    assert all(isinstance(content, str) for content in returns.values())


def test_tools_report_missing_data():
    model = scripted_model([("describe_data", {}), ("create_histogram", {"column": "x"})])
    with analyst_agent.override(model=model):
        _, messages, _, _ = asyncio.run(run_analyst("describe", df=None))
    returns = _tool_returns(messages)
    assert returns["describe_data"] == NO_DATA_MESSAGE
    assert returns["create_histogram"] == NO_DATA_MESSAGE


def test_tool_errors_are_returned_to_the_model(iris):
    model = scripted_model([("run_pca", {"columns": "not_a_column"})])
    with analyst_agent.override(model=model):
        _, messages, _, _ = asyncio.run(run_analyst("pca", df=iris, plots=[]))
    assert "Columns not found" in _tool_returns(messages)["run_pca"]


def test_segmentation_workflow_updates_dataframe_and_panels(iris):
    features = "sepal_length,sepal_width,petal_length,petal_width"
    model = scripted_model([
        ("find_optimal_clusters", {"columns": features, "max_k": 6}),
        ("run_kmeans", {"columns": features, "n_clusters": 3}),
        ("run_pca", {"columns": features}),
        ("create_pca_plot", {"color_by": "cluster"}),
        ("record_finding", {"title": "Three segments", "description": "...", "importance": "high"}),
    ])
    state = AnalysisState()
    plots: list = []
    with analyst_agent.override(model=model):
        response, messages, new_plots, df = asyncio.run(
            run_analyst("segment", df=iris, state=state, plots=plots)
        )

    returns = _tool_returns(messages)
    assert "Silhouette score" in returns["run_kmeans"]
    assert "Colored by: cluster" in returns["create_pca_plot"]
    assert response == "done"
    assert len(new_plots) == 2  # elbow plot + PCA scatter
    assert {"cluster", "PC1", "PC2"} <= set(get_derived_columns(df))
    assert [f.title for f in state.findings] == ["Three segments"]
