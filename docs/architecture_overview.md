# CyberStat — Detailed Architecture Overview

> A complete walkthrough of how the CyberStat data analytics agent works,  
> from user interaction to LLM tool calls and back.

---

## Table of Contents

1. [30-Second Summary](#30-second-summary)
2. [Layer 1: Streamlit UI (app.py)](#layer-1-streamlit-ui-apppy)
3. [Layer 2: Inputs — How Data Enters the System](#layer-2-inputs--how-data-enters-the-system)
4. [Layer 3: The Bridge (get_agent_response)](#layer-3-the-bridge-get_agent_response)
5. [Layer 4: The PydanticAI Agent (agent/analyst.py)](#layer-4-the-pydanticai-agent-agentanalystpy)
   - [Dependencies (AnalysisDependencies)](#concept-1-dependencies-analysisdependencies)
   - [Agent Creation](#concept-2-agent-creation)
   - [Tool Registration (@agent.tool)](#concept-3-tool-registration-agenttool)
   - [Tools with Parameters](#concept-4-tools-with-parameters)
   - [Running the Agent (run_analyst)](#concept-5-running-the-agent-run_analyst)
   - [RunContext Explained](#concept-6-runcontextanalysisdependencies-explained)
6. [Layer 5: Tool Implementations (agent/tools/)](#layer-5-tool-implementations-agenttools)
7. [Layer 6: Pydantic Models (agent/models/)](#layer-6-pydantic-models-agentmodels)
8. [Layer 7: UI Rendering (components/)](#layer-7-ui-rendering-components)
9. [Complete Data Flow Diagram](#complete-data-flow-diagram)
10. [Design Pattern: Tool-Augmented LLM Agent (ReAct)](#design-pattern-tool-augmented-llm-agent-react)
11. [File Map](#file-map)

---

## 30-Second Summary

User uploads a CSV in Streamlit, types a question, Streamlit calls the PydanticAI agent,
the agent decides which tools to call (stats, viz, ML), tools operate on a shared DataFrame
and write results to shared state, Streamlit re-renders everything.

---

## Layer 1: Streamlit UI (`app.py`)

This is the **entry point** — the only file Streamlit executes directly. Everything else is imported.

### Startup Sequence

Streamlit re-executes `app.py` **top-to-bottom on every user interaction** (click, submit, upload).
That's why session state is critical.

```python
st.set_page_config(
    page_title="CyberStat - Data Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

Then a CSS block injects the cyberpunk theme (pure visual, no logic).

### Session State: The "Database" Between Reruns

```python
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []             # Chat history (list[ChatMessage])
    if "analysis_state" not in st.session_state:
        st.session_state.analysis_state = AnalysisState()  # Side panel state
    if "df" not in st.session_state:
        st.session_state.df = None                 # The loaded DataFrame
    if "pydantic_messages" not in st.session_state:
        st.session_state.pydantic_messages = None  # PydanticAI message history (agent memory)
    if "plots" not in st.session_state:
        st.session_state.plots = []                # Generated viz: [(bytes, description), ...]
```

There are **5 pieces of persistent state**:

| Key                | Type               | Purpose                                              |
|--------------------|--------------------|------------------------------------------------------|
| `messages`         | `list[ChatMessage]`| What the user sees in the chat window                |
| `analysis_state`   | `AnalysisState`    | Findings, recommendations, activity log, data summary|
| `df`               | `pd.DataFrame`     | The actual data being analyzed                       |
| `pydantic_messages`| `list`             | PydanticAI's internal message history (LLM memory)   |
| `plots`            | `list[(bytes,str)]`| Generated visualization images + descriptions        |

### Two-Column Layout

```python
chat_col, panel_col = st.columns([2, 1], gap="medium")
```

- **Left column (2/3 width):** Chat interface + Visualizations grid
- **Right column (1/3 width):** File upload + Analysis panel (summary, findings, recommendations, log)

---

## Layer 2: Inputs — How Data Enters the System

There are exactly **two inputs**:

### Input A: File Upload

```python
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])
```

The flow:
1. `st.file_uploader` receives a CSV/Excel file
2. `process_uploaded_file()` reads it into a `pd.DataFrame`
3. `create_data_summary()` creates a `DataSummary` Pydantic model (rows, columns, types, missing values)
4. Both are stored in `st.session_state`
5. `st.rerun()` triggers a full re-render so everything updates

A `file_key` (`name_size`) is used to detect re-uploads vs. the same file:

```python
file_key = f"{uploaded_file.name}_{uploaded_file.size}"
if "current_file" not in st.session_state or st.session_state.current_file != file_key:
    # New file — process it
```

### Input B: Chat Message

```python
user_input = st.chat_input("Ask about your data...")

if user_input:
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    with st.spinner("Analyzing..."):
        response = asyncio.run(get_agent_response(user_input))
    st.session_state.messages.append(ChatMessage(role="assistant", content=response))
    st.rerun()
```

This is the **critical handoff point** — `get_agent_response()` bridges the UI to the AI agent.

---

## Layer 3: The Bridge (`get_agent_response`)

```python
async def get_agent_response(user_message: str) -> str:
    new_plots = []

    response, message_history, plots, updated_df = await run_analyst(
        user_message=user_message,
        df=st.session_state.df,
        state=st.session_state.analysis_state,
        message_history=st.session_state.pydantic_messages,
        plots=new_plots
    )

    st.session_state.pydantic_messages = message_history
    if plots:
        st.session_state.plots.extend(plots)
    if updated_df is not None:
        st.session_state.df = updated_df

    return response
```

This function **packs 4 things into `run_analyst`** and **unpacks 4 things back**:

```
         INPUTS                           OUTPUTS
  ┌─────────────────┐              ┌─────────────────────┐
  │  user_message    │──────┐      │  response (str)     │ → chat bubble
  │  df (DataFrame)  │      │      │  message_history    │ → pydantic_messages (memory)
  │  analysis_state  │      ├─────>│  plots (list)       │ → visualization grid
  │  message_history │      │      │  updated_df         │ → may have new columns (PCA/kmeans)
  │  new_plots []    │──────┘      └─────────────────────┘
  └─────────────────┘
```

> **Note:** `analysis_state` is a mutable object. Tools modify it **in place** (adding findings,
> recommendations, log entries). It doesn't need to be "returned" — the same object in
> `st.session_state` gets updated by reference.

---

## Layer 4: The PydanticAI Agent (`agent/analyst.py`)

This is the heart of the application. It contains all the PydanticAI concepts.

### Concept 1: Dependencies (`AnalysisDependencies`)

```python
@dataclass
class AnalysisDependencies:
    df: Optional[pd.DataFrame] = None
    state: Optional[AnalysisState] = None
    plots: Optional[list] = None  # List of (image_bytes, description) tuples
```

This is PydanticAI's **dependency injection pattern**. Instead of global variables, you bundle
everything the agent's tools need into a single typed object. Every tool receives this via `RunContext`.

**Why a dataclass, not a Pydantic model?** Dependencies are a runtime container holding mutable
references (a DataFrame, a list). They don't need validation or serialization.

### Concept 2: Agent Creation

```python
analyst_agent = Agent(
    model="google-gla:gemini-3-pro-preview",
    system_prompt=SYSTEM_PROMPT,
    deps_type=AnalysisDependencies,
)
```

Three things happen here:

1. **`model`** — which LLM to use. `google-gla:` tells PydanticAI to use Google/Gemini.
2. **`system_prompt`** — instructions shaping agent behavior. Sent with every LLM call.
3. **`deps_type`** — tells PydanticAI what type dependencies will be, enabling type-checking.

The agent is created **once at module import time** (not per-request). It's stateless — all
mutable state lives in the dependencies.

### Concept 3: Tool Registration (`@agent.tool`)

Every function decorated with `@analyst_agent.tool` becomes a tool the LLM can call:

```python
@analyst_agent.tool
def describe_data(ctx: RunContext[AnalysisDependencies]) -> str:
    """
    Get a comprehensive statistical summary of the loaded dataset.
    Use this to understand the overall structure and distributions in the data.
    """
    if ctx.deps.df is None:
        return "No data loaded. Please ask the user to upload a dataset first."

    if ctx.deps.state:
        ctx.deps.state.add_activity("Generated data summary", "Running describe() on all columns")

    return tools.describe_dataframe(ctx.deps.df)
```

**How PydanticAI uses this:**

1. The **function name** (`describe_data`) becomes the tool name the LLM sees
2. The **docstring** becomes the tool description the LLM reads to decide when to use it
3. The **parameters** (besides `ctx`) become the tool's input schema — PydanticAI auto-generates
   a JSON schema from the type hints and sends it to the LLM
4. The **return value** (always `str`) goes back to the LLM as the tool result

The flow:

```
LLM sees:    "describe_data: Get a comprehensive statistical summary..."
LLM decides: "I should call describe_data"
LLM outputs: tool_call(name="describe_data", args={})
PydanticAI:  calls describe_data(ctx=RunContext(deps=...))
Tool returns: "Statistical Summary:\n  age  salary..."
PydanticAI:  sends this string back to the LLM
LLM:         uses the stats to write a natural language response
```

### Concept 4: Tools with Parameters

When a tool has extra parameters, PydanticAI generates a JSON schema for the LLM:

```python
@analyst_agent.tool
def compare_groups(
    ctx: RunContext[AnalysisDependencies],  # ← hidden from LLM, injected by PydanticAI
    numeric_column: str,                     # ← visible to LLM as {"type": "string"}
    group_column: str                        # ← visible to LLM as {"type": "string"}
) -> str:
    """
    Compare a numeric variable across two groups using a t-test.

    Args:
        numeric_column: The numeric column to compare (e.g., 'salary', 'score')
        group_column: The categorical column defining groups (must have exactly 2 unique values)
    """
```

The LLM sees a tool with two string parameters. It uses the `Args` descriptions plus its
knowledge of the dataset (from previous tool calls) to choose the right column names.
PydanticAI handles `ctx` behind the scenes — the LLM never knows it exists.

### Concept 5: Running the Agent (`run_analyst`)

```python
async def run_analyst(user_message, df, state, message_history, plots):
    new_plots = [] if plots is None else plots
    deps = AnalysisDependencies(df=df, state=state, plots=new_plots)

    result = await analyst_agent.run(
        user_message,
        deps=deps,
        message_history=message_history
    )

    return result.output, result.all_messages(), new_plots, deps.df
```

`analyst_agent.run()` is a single call that orchestrates the entire conversation turn:

1. Sends system prompt + message history + user message to Gemini
2. Gemini responds (possibly with tool calls)
3. PydanticAI executes each tool call, injecting `ctx` with `deps`
4. Sends tool results back to Gemini
5. Gemini responds again (possibly calling more tools, or producing a final answer)
6. Repeats steps 2–5 until Gemini produces a text-only response
7. Returns a `RunResult` with `.output` (final text) and `.all_messages()` (full conversation)

**The `message_history` parameter is what gives the agent memory.** Each call returns the full
message list, stored in `st.session_state.pydantic_messages` and passed back on the next call.

### Concept 6: `RunContext[AnalysisDependencies]` Explained

`RunContext` is a PydanticAI class that acts as a **delivery envelope**. When the LLM calls a
tool, PydanticAI needs a way to give that tool access to your application's data. `RunContext`
is that mechanism.

The `[AnalysisDependencies]` part is a **generic type parameter** — it tells Python that the
`.deps` attribute will be an `AnalysisDependencies` instance.

**The connection chain:**

```
1. You define what tools need:
   @dataclass
   class AnalysisDependencies:
       df, state, plots

2. You tell the Agent about it:
   Agent(..., deps_type=AnalysisDependencies)

3. At runtime, you create an instance and pass it to .run():
   deps = AnalysisDependencies(df=df, state=state, plots=[])
   agent.run(message, deps=deps)

4. PydanticAI wraps it in RunContext and injects it into each tool:
   def describe_data(ctx: RunContext[AnalysisDependencies]) -> str:
       ctx.deps.df      → your DataFrame
       ctx.deps.state   → your AnalysisState
       ctx.deps.plots   → your plots list
```

**Visual diagram:**

```
app.py                          PydanticAI internals              your tool function
───────                         ──────────────────                ──────────────────

deps = AnalysisDependencies(
    df=dataframe,                     │
    state=analysis_state,             │
    plots=[]                          │
)                                     │
                                      ▼
agent.run(msg, deps=deps) ──► wraps deps in RunContext ──► describe_data(ctx)
                                      │                           │
                                      │                           ├─ ctx.deps.df
                                      │                           ├─ ctx.deps.state
                                      │                           └─ ctx.deps.plots
                                      │
                               also hides `ctx` from the
                               JSON schema sent to Gemini
```

**Why not just use global variables?** `RunContext` solves three problems:

1. **No globals** — each call gets its own isolated `deps`. Multiple concurrent analyses
   won't stomp on each other.
2. **Type safety** — your IDE gives you autocomplete on `ctx.deps.df`, `ctx.deps.state`, etc.
3. **Invisible to the LLM** — PydanticAI strips `ctx` from the tool schema so the LLM only
   sees the parameters it should provide (like `column_name: str`).

---

## Layer 5: Tool Implementations (`agent/tools/`)

Tools are organized into three tiers:

### Tier A: Pure Stat Functions (stateless)

These take a DataFrame and return a formatted string. No side effects.

| File                 | Function                          | What It Does                              |
|----------------------|-----------------------------------|-------------------------------------------|
| `describe.py`        | `describe_dataframe(df)`          | `df.describe()` formatted as string       |
| `column_info.py`     | `get_column_info(df, col)`        | Type, missing, stats, outlier count       |
| `correlation.py`     | `calculate_correlation(df, method)` | Correlation matrix + notable pairs      |
| `ttest.py`           | `perform_ttest(df, col, group_col)` | Independent t-test with Cohen's d       |
| `normality.py`       | `check_normality(df, col)`        | Shapiro-Wilk test                         |
| `value_counts.py`    | `get_value_counts(df, col, top_n)` | Frequency table                          |
| `group_statistics.py`| `get_group_statistics(df, num, grp)` | Per-group mean/median/std/IQR + comparison |

### Tier B: ML Functions (return modified DataFrame + stats)

These return `tuple[pd.DataFrame, dict]` — a new DataFrame with added columns and a stats dict.

| File        | Function                        | Returns                                    |
|-------------|---------------------------------|--------------------------------------------|
| `pca.py`    | `perform_pca(df, columns, n)`   | df + PC1, PC2... columns; variance stats   |
| `kmeans.py` | `perform_kmeans(df, columns, k)` | df + `cluster` column; cluster profiles   |
| `kmeans.py` | `find_optimal_k(df, cols, max)` | dict with inertia curve + suggested k      |

**Key pattern:** These tools mutate `ctx.deps.df` in the analyst wrapper:

```python
ctx.deps.df = result_df  # PCA columns / cluster labels now available for next tools
```

That's how PCA columns become available for clustering, and cluster labels become available
for visualization. The modified `deps.df` flows back to `app.py` via `run_analyst`.

### Tier C: Visualization Functions (return image bytes + stats)

Every viz function returns `tuple[bytes, str, dict]`:

- `bytes` — PNG image data
- `str` — description text
- `dict` — statistics extracted from the plot

| File              | Function                      | Visualization Type              |
|-------------------|-------------------------------|---------------------------------|
| `histogram.py`    | `create_histogram()`          | Distribution of a numeric column|
| `scatter.py`      | `create_scatter_plot()`       | Relationship between two columns|
| `heatmap.py`      | `create_correlation_heatmap()`| All pairwise correlations       |
| `boxplot.py`      | `create_box_plot()`           | Distribution, optionally grouped|
| `bar_chart.py`    | `create_bar_chart()`          | Category frequencies            |
| `pca_plot.py`     | `create_pca_scatter()`        | PCA 2D projection               |
| `pca_plot.py`     | `create_scree_plot()`         | Variance explained per component|
| `cluster_plot.py` | `create_cluster_scatter()`    | Cluster assignments on 2D space |
| `cluster_plot.py` | `create_elbow_plot()`         | Optimal k selection             |
| `cluster_plot.py` | `create_cluster_profile_chart()` | Feature means by cluster     |

The analyst tool wrappers append `(bytes, description)` to `ctx.deps.plots`, then return the
**stats as a string** to the LLM. The LLM never "sees" the image — it gets the numbers the
chart represents, ensuring its text matches the visualization.

### Visualization Theming

All plots share a consistent cyberpunk theme defined in `agent/tools/visualizations/__init__.py`:

- `THEME` / `CYBER_COLORS` — color constants (neon blue, pink, purple, etc.)
- `CYBER_PALETTE` — 8-color cycle for data series
- `matplotlib.rcParams` — global plot styling (dark background, grid, fonts)
- `format_label()` — converts `snake_case` column names to "Title Case" labels

The `matplotlib.use('Agg')` call at the top forces a non-interactive backend, which is required
for server-side rendering (no display window).

---

## Layer 6: Pydantic Models (`agent/models/`)

These are the data contracts:

```
AnalysisState          (the root container)
├── data_summary: DataSummary            → file info, column types, missing values
├── findings: list[Finding]              → title, description, importance, columns
├── recommendations: list[Recommendation]→ title, description, priority
└── activity_log: list[ActivityLogEntry] → timestamp, action, details
```

Plus `ChatMessage` (role + content + timestamp) for the UI chat history.

### Model Details

| Model              | Fields                                           | Used By              |
|--------------------|--------------------------------------------------|----------------------|
| `AnalysisState`    | `data_summary`, `findings`, `recommendations`, `activity_log` | Side panel, all tools |
| `DataSummary`      | `filename`, `rows`, `columns`, `column_names`, `numeric_columns`, `categorical_columns`, `missing_values` | File upload |
| `Finding`          | `title`, `description`, `importance`, `related_columns` | `record_finding` tool |
| `Recommendation`   | `title`, `description`, `priority`               | `record_recommendation` tool |
| `ActivityLogEntry` | `timestamp`, `action`, `details`                 | Every tool via `state.add_activity()` |
| `ChatMessage`      | `role`, `content`, `timestamp`                   | Chat UI              |

**Why Pydantic?** These models enforce types at construction time. If the agent produced invalid
data (wrong field type), Pydantic catches it. They also make serialization trivial if you ever
want to save/load sessions.

---

## Layer 7: UI Rendering (`components/`)

### `chat.py`

```python
def render_chat_history(messages: list[ChatMessage]) -> None:
    for msg in messages:
        with st.chat_message(msg.role, avatar=avatar):
            st.markdown(escape_latex(msg.content))
```

Pure rendering. Reads the `messages` list, draws chat bubbles.
`escape_latex()` prevents `$` in data values from triggering Streamlit's LaTeX renderer.

### `analysis_panel.py`

Reads `AnalysisState` and renders four sections:

1. **Data Summary** — metrics (rows/columns), column type lists, missing value warnings
2. **Key Findings** — expandable cards, color-coded by importance (🔴 high, 🟡 medium, 🟢 low)
3. **Recommendations** — priority-badged list (`:red[HIGH]`, `:orange[MEDIUM]`, `:green[LOW]`)
4. **Activity Log** — collapsible, reverse-chronological feed of tool actions

---

## Complete Data Flow Diagram

Example: user types "Are age and salary correlated?"

```
User types message
       │
       ▼
   app.py: st.chat_input
       │
       ▼
   app.py: get_agent_response(user_message)
       │
       ├── Packs: df, state, message_history, plots[]
       │
       ▼
   analyst.py: run_analyst()
       │
       ├── Creates AnalysisDependencies(df, state, plots)
       │
       ▼
   analyst.py: analyst_agent.run(message, deps, history)
       │
       ▼
   PydanticAI ──► Gemini API
       │
       │  Gemini decides: "I need get_correlations and create_scatter"
       │
       │  ┌─ Tool call 1: get_correlations(method="pearson")
       │  │   → analyst.py wrapper → tools.calculate_correlation(df)
       │  │   → returns: "Correlation Matrix:\n  age & salary: 0.85 (strong positive)"
       │  │   → state.add_activity("Calculated pearson correlations")
       │  │
       │  ├─ Tool call 2: create_scatter(x="age", y="salary")
       │  │   → analyst.py wrapper → tools.create_scatter_plot(df, "age", "salary")
       │  │   → returns: (png_bytes, "Scatter of salary vs age", {r: 0.85, ...})
       │  │   → plots.append((png_bytes, description))
       │  │   → state.add_activity("Created scatter plot")
       │  │   → returns stats string to LLM
       │  │
       │  ├─ Tool call 3: record_finding(title="Strong age-salary correlation", ...)
       │  │   → state.add_finding(...)
       │  │
       │  └─ Gemini produces final text using the stats it received
       │
       ▼
   Returns: (response_text, messages, plots, df)
       │
       ▼
   app.py: get_agent_response unpacks:
       ├── st.session_state.pydantic_messages = messages  (memory for next call)
       ├── st.session_state.plots.extend(plots)           (new charts)
       ├── st.session_state.df = updated_df               (if PCA/kmeans added columns)
       └── returns response_text
       │
       ▼
   app.py: messages.append(ChatMessage("assistant", response_text))
   app.py: st.rerun()
       │
       ▼
   Full page re-renders:
       ├── Left col: chat.py renders all messages (including new one)
       ├── Left col: visualizations grid shows new scatter plot
       └── Right col: analysis_panel.py shows new finding + activity log entries
```

---

## Design Pattern: Tool-Augmented LLM Agent (ReAct)

The overall pattern is called **ReAct** (Reason + Act):

1. **Reason**: The LLM reads the user's question and decides what information it needs
2. **Act**: It calls tools to get that information
3. **Observe**: It reads the tool results
4. **Repeat**: It may call more tools based on what it learned
5. **Respond**: When it has enough info, it produces a final natural-language answer

PydanticAI abstracts steps 2–4 into a loop inside `agent.run()`. You define tools and the
framework handles the orchestration.

### The Four PydanticAI Primitives Used

| Primitive                              | Your Code             | Purpose                                      |
|----------------------------------------|-----------------------|----------------------------------------------|
| `Agent(model, system_prompt, deps_type)` | `analyst_agent`     | Configures the LLM + persona + dependency type |
| `@agent.tool`                          | 20+ decorated funcs   | Registers callable tools the LLM can invoke  |
| `RunContext[T]`                        | `ctx` in every tool   | Injects shared state into each tool call     |
| `agent.run(msg, deps, history)`        | Inside `run_analyst()`| Runs the reason-act-observe loop             |

The `message_history` parameter enables **multi-turn conversations** — without it, each call
would be stateless and the agent would forget prior context.

---

## File Map

```
CyberStat/
├── app.py                          # Streamlit entry point, layout, session state
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── .gitignore                      # Secrets and build artifacts excluded
├── README.md                       # User-facing documentation
│
├── agent/
│   ├── __init__.py                 # Exports: analyst_agent, run_analyst, AnalysisDependencies
│   ├── analyst.py                  # THE CORE: Agent, deps, system prompt, 20+ tool wrappers
│   │
│   ├── models/
│   │   ├── __init__.py             # Re-exports all models
│   │   ├── activity_log.py         # ActivityLogEntry
│   │   ├── analysis_state.py       # AnalysisState (root container)
│   │   ├── chat_message.py         # ChatMessage
│   │   ├── data_summary.py         # DataSummary
│   │   ├── finding.py              # Finding
│   │   └── recommendation.py       # Recommendation
│   │
│   └── tools/
│       ├── __init__.py             # Re-exports all tool functions
│       ├── column_info.py          # Single-column analysis
│       ├── correlation.py          # Correlation matrix
│       ├── describe.py             # df.describe() wrapper
│       ├── group_statistics.py     # Per-group stats comparison
│       ├── kmeans.py               # K-means clustering + elbow method
│       ├── normality.py            # Shapiro-Wilk test
│       ├── pca.py                  # Principal Component Analysis
│       ├── ttest.py                # Independent samples t-test
│       ├── value_counts.py         # Frequency distribution
│       │
│       └── visualizations/
│           ├── __init__.py         # Theme config, CYBER_COLORS, format_label()
│           ├── bar_chart.py        # Horizontal bar chart
│           ├── boxplot.py          # Box plot (single or grouped)
│           ├── cluster_plot.py     # Cluster scatter, elbow plot, profile chart
│           ├── heatmap.py          # Correlation heatmap
│           ├── histogram.py        # Histogram with mean/median lines
│           ├── pca_plot.py         # PCA scatter, scree plot
│           └── scatter.py          # Scatter plot with trend line
│
├── components/
│   ├── __init__.py
│   ├── analysis_panel.py           # Right-side panel renderer
│   └── chat.py                     # Chat history renderer
│
├── data/                           # Sample datasets for testing
│   ├── customer_data.csv
│   ├── employee_data.csv
│   ├── fitness_data.csv
│   ├── iris.csv
│   ├── sales_data.csv
│   └── survey_data.csv
│
├── docs/
│   ├── detailed_architecture_overview.md   # ← This file
│   └── future_graph_architecture.md
│
└── Screenshots/
    └── example.png
```
