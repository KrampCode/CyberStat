# CyberStat v2: Graph Architecture Ideas

> Future enhancement ideas for converting CyberStat from a single-agent to a multi-agent graph architecture using `pydantic-graph`.

---

## Current Architecture (v1)

Single "generalist" agent with all tools:
- 6 statistical analysis tools
- 5 visualization tools
- 2 insight recording tools

**Pros:** Simple, works well for linear conversations
**Cons:** Large prompt, all-or-nothing execution, hard to extend

---

## Proposed Graph Architecture (v2)

### High-Level Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Coordinator │ ──▶ │ Data Quality │ ──▶ │ Analysis    │ ──▶ │ Report     │
│ (Router)    │     │ Check        │     │ Phase       │     │ Generation │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
                                               │
                           ┌───────────────────┼───────────────────┐
                           ▼                   ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                    │ Statistics  │     │ Visualiza-  │     │ Insights    │
                    │ Agent       │     │ tion Agent  │     │ Agent       │
                    └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Specialized Agents

### 1. Coordinator Agent
**Purpose:** Route queries to the appropriate specialist

**System Prompt Focus:**
- Classify user intent (new data, analysis question, visualization request, summary)
- Decide which agents to invoke and in what order

**Tools:**
- `classify_intent()` - Determine what the user wants
- `select_agents()` - Choose which specialists to call

**Output:** Routing decision + execution plan

---

### 2. Data Quality Agent
**Purpose:** Check data quality before analysis

**System Prompt Focus:**
- Expert at finding data issues
- Conservative - flag potential problems

**Tools:**
- `check_missing_values()` - Find and quantify missing data
- `detect_outliers()` - Statistical outlier detection
- `validate_data_types()` - Check for type mismatches
- `check_duplicates()` - Find duplicate rows

**Output:** Quality report with warnings/recommendations

---

### 3. Statistics Agent
**Purpose:** Perform statistical analyses

**System Prompt Focus:**
- Deep statistical expertise
- Choose appropriate tests based on data characteristics

**Tools:**
- `describe_data()` - Basic statistics
- `calculate_correlation()` - Correlation analysis
- `perform_ttest()` - Group comparisons
- `check_normality()` - Distribution tests
- `run_regression()` - (future) Linear regression

**Output:** Statistical findings with interpretations

---

### 4. Visualization Agent
**Purpose:** Create appropriate visualizations

**System Prompt Focus:**
- Choose the right chart for the data/question
- Create clear, informative plots

**Tools:**
- `create_histogram()` - Distribution plots
- `create_scatter()` - Relationship plots
- `create_heatmap()` - Correlation overview
- `create_boxplot()` - Distribution comparison
- `create_bar_chart()` - Categorical data

**Output:** Generated plots with descriptions

---

### 5. Insights Agent
**Purpose:** Synthesize findings into actionable insights

**System Prompt Focus:**
- Connect statistical findings to business meaning
- Prioritize findings by importance

**Tools:**
- `record_finding()` - Save key insights
- `record_recommendation()` - Save action items
- `connect_findings()` - Link related discoveries

**Output:** Prioritized insights and recommendations

---

### 6. Report Agent
**Purpose:** Create user-friendly final response

**System Prompt Focus:**
- Clear communication
- Appropriate level of technical detail

**Tools:**
- `format_response()` - Structure the output
- `generate_summary()` - Create executive summary
- `evaluate_quality()` - Check response completeness

**Output:** Final response to user

---

## State Object

```python
@dataclass
class GraphState:
    # Input
    df: pd.DataFrame
    user_query: str
    
    # Routing
    intent: str  # "new_data", "analysis", "visualization", "summary"
    execution_plan: list[str]  # Agents to call
    
    # Quality Phase
    quality_report: QualityReport | None
    
    # Analysis Phase
    stat_findings: list[StatFinding]
    visualizations: list[Plot]
    
    # Synthesis Phase
    insights: list[Insight]
    recommendations: list[Recommendation]
    
    # Output
    final_response: str
    
    # Control
    refinement_count: int = 0
    max_refinements: int = 2
```

---

## Conditional Routing Examples

### Route by Data Type
```
Coordinator detects:
├── Time series data → Time Series Agent (seasonal, trends)
├── Categorical heavy → Categorical Agent (chi-square, frequencies)
├── Numeric heavy → Statistics Agent (correlations, t-tests)
└── Mixed → General Analysis Agent
```

### Route by User Intent
```
User query classified as:
├── "What's in this data?" → Quality + Stats + Summary
├── "Is X correlated with Y?" → Stats only (focused)
├── "Show me a chart of X" → Visualization only
├── "Give me a full report" → All agents in sequence
└── "Clean up the data" → Quality + Recommendations
```

---

## Human-in-the-Loop Checkpoints

Add pause points for user confirmation:

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│ Initial     │ ──▶ │ User Confirms   │ ──▶ │ Deep Dive   │
│ Scan        │     │ Focus Areas     │     │ Analysis    │
└─────────────┘     └─────────────────┘     └─────────────┘
```

**Use cases:**
- Confirm which columns to analyze (avoid wasting compute)
- Approve data cleaning operations before applying
- Select which findings to explore deeper

---

## Refinement Loop

Allow the graph to iterate until quality threshold met:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Generate    │ ──▶ │ Evaluate    │ ──▶ │ Good        │
│ Response    │     │ Quality     │     │ Enough?     │
└─────────────┘     └─────────────┘     └──────┬──────┘
       ▲                                       │
       │                                  No   │  Yes
       └───────────────────────────────────────┘   │
                                                   ▼
                                            ┌─────────────┐
                                            │ Return to   │
                                            │ User        │
                                            └─────────────┘
```

**Quality criteria:**
- All user questions addressed?
- Sufficient statistical backing?
- Visualizations included where helpful?
- Recommendations actionable?

---

## Cost Optimization Strategy

Use different models for different agents:

| Agent | Model | Reasoning |
|-------|-------|-----------|
| Coordinator | gemini-flash (cheap) | Simple routing decisions |
| Quality | gemini-flash | Rule-based checks |
| Statistics | gemini-pro | Complex interpretation |
| Visualization | gemini-flash | Straightforward tool calls |
| Insights | gemini-pro | Requires synthesis |
| Report | gemini-pro | User-facing quality matters |

**Estimated cost reduction:** 40-60% vs using pro for everything

---

## Implementation Steps

### Phase 1: Add Quality Check Node
1. Create `QualityAgent` with data validation tools
2. Run automatically when new data is uploaded
3. Surface warnings before analysis

### Phase 2: Split Analysis Agents
1. Separate Statistics and Visualization agents
2. Coordinator decides which to call based on query
3. Keep single prompt per agent (focused)

### Phase 3: Add Insights Synthesis
1. Create `InsightsAgent` that receives findings from other agents
2. Synthesizes into prioritized insights
3. Generates recommendations

### Phase 4: Add Report Generation
1. Create `ReportAgent` for final formatting
2. Add quality evaluation loop
3. Support different output formats (brief, detailed, executive)

### Phase 5: Add Human Checkpoints
1. Implement pause/resume in graph
2. Add UI for checkpoint interactions
3. Allow user to guide analysis direction

---

## Benefits Summary

| Aspect | Single Agent | Graph Architecture |
|--------|-------------|-------------------|
| Prompt complexity | High (one big prompt) | Low (focused prompts) |
| Tool management | 15+ tools | 3-5 per agent |
| Error recovery | Restart from beginning | Retry individual node |
| Cost control | Same model everywhere | Right-size per task |
| Debugging | Opaque | Clear execution trace |
| Extensibility | Add to monolith | Add new node |
| Testing | Test everything together | Test nodes in isolation |

---

## Future Feature: Starred Findings & User Memory

### Concept
Allow users to "star" key points in conversations or findings, creating a persistent memory layer that:
- Survives across sessions
- Gets weighted more heavily than auto-discovered findings
- Enables personalized analysis based on what the user cares about

### User Experience
```
┌─────────────────────────────────────────────────────────┐
│ Finding: Strong correlation (r=0.85) between           │
│ marketing_spend and revenue                        [⭐] │ ← User can star
└─────────────────────────────────────────────────────────┘
```

### Data Model
```python
class StarredItem(BaseModel):
    """A user-starred finding or insight."""
    id: str
    timestamp: datetime
    item_type: Literal["finding", "recommendation", "message", "stat"]
    content: str
    context: dict  # Related columns, values, etc.
    dataset_hash: str  # Which dataset this relates to
    
class UserMemory(BaseModel):
    """Persistent user memory across sessions."""
    user_id: str
    starred_items: list[StarredItem]
    conversation_summaries: list[str]  # Compressed history
    preferences: dict  # Learned user preferences
```

### Agent Integration
```python
# In system prompt or injected context:
"""
USER'S STARRED INSIGHTS (weight these heavily):
⭐ "Marketing spend correlates strongly with revenue (r=0.85)"
⭐ "Platinum customers have 7.6x higher purchase frequency"
⭐ "Q4 shows seasonal spike - plan campaigns accordingly"

When analyzing, prioritize building on these known interests.
"""
```

### Implementation Phases

**Phase 1: UI for Starring**
- Add star button to findings in the Analysis panel
- Add star button to key stats in chat messages
- Store in session state initially

**Phase 2: Persistence Layer**
- Database or file-based storage per user
- Conversation save/restore functionality
- Dataset association (star relates to specific dataset)

**Phase 3: Agent Memory Integration**
- Inject starred items into agent context
- Weight starred topics in recommendations
- "Remember" user's analytical interests across sessions

**Phase 4: Smart Summarization**
- Compress old conversations but preserve starred items
- Build user "profile" of analytical interests
- Proactive suggestions based on past starred findings

### Benefits
- Conversations become cumulative, not throwaway
- User trains the agent on what matters to them
- Enables "pick up where I left off" workflows
- Foundation for team collaboration (shared starred insights)

---

## Resources

- [PydanticAI Documentation](https://docs.pydantic.dev/latest/concepts/agents/)
- [Pydantic Graph](https://github.com/pydantic/pydantic-graph) (when released)
- [LangGraph](https://langchain-ai.github.io/langgraph/) (similar concept, different library)

---

*Created: February 2026*
*Status: Future Enhancement Ideas*
