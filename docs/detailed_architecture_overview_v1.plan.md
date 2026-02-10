# CyberStat — Detailed Architecture & Design Decisions (v1)

> A deep-dive into every major construct, function pattern, and design choice  
> in the CyberStat codebase. This goes beyond "what it does" into "why it's  
> built this way" and "how each piece is constructed."

---

## Table of Contents

1. [Overall Design Philosophy](#1-overall-design-philosophy)
2. [The Streamlit Execution Model & Why It Matters](#2-the-streamlit-execution-model--why-it-matters)
3. [Session State Architecture](#3-session-state-architecture)
4. [The Async Bridge: nest_asyncio + asyncio.run()](#4-the-async-bridge-nest_asyncio--asynciorun)
5. [PydanticAI — Every Pattern In Detail](#5-pydanticai--every-pattern-in-detail)
   - 5a. [Agent Construction](#5a-agent-construction)
   - 5b. [The Dependencies Dataclass](#5b-the-dependencies-dataclass)
   - 5c. [RunContext — The Injection Mechanism](#5c-runcontext--the-injection-mechanism)
   - 5d. [Tool Registration — The @agent.tool Decorator](#5d-tool-registration--the-agenttool-decorator)
   - 5e. [Tool Signatures — What the LLM Sees vs. What It Doesn't](#5e-tool-signatures--what-the-llm-sees-vs-what-it-doesnt)
   - 5f. [agent.run() — The Orchestration Loop](#5f-agentrun--the-orchestration-loop)
   - 5g. [RunResult — Extracting Outputs](#5g-runresult--extracting-outputs)
   - 5h. [Message History — Multi-Turn Memory](#5h-message-history--multi-turn-memory)
6. [System Prompt Engineering](#6-system-prompt-engineering)
7. [Tool Construction Patterns](#7-tool-construction-patterns)
   - 7a. [The Three-Layer Tool Pattern](#7a-the-three-layer-tool-pattern)
   - 7b. [Statistical Tool Functions](#7b-statistical-tool-functions)
   - 7c. [Visualization Tool Functions](#7c-visualization-tool-functions)
   - 7d. [ML Tool Functions (PCA & K-Means)](#7d-ml-tool-functions-pca--k-means)
   - 7e. [State-Writing Tools (Findings & Recommendations)](#7e-state-writing-tools-findings--recommendations)
8. [The Visualization Paradigm](#8-the-visualization-paradigm)
   - 8a. [The Return Signature Convention](#8a-the-return-signature-convention)
   - 8b. [The Theme System](#8b-the-theme-system)
   - 8c. [The format_label() Utility](#8c-the-format_label-utility)
   - 8d. [The BytesIO Pipeline](#8d-the-bytesio-pipeline)
   - 8e. [Stats-Alongside-Image Pattern](#8e-stats-alongside-image-pattern)
9. [Pydantic Models — The Data Contracts](#9-pydantic-models--the-data-contracts)
10. [The File Upload Pipeline](#10-the-file-upload-pipeline)
11. [The Visualization Grid Renderer](#11-the-visualization-grid-renderer)
12. [Package Structure & Import Strategy](#12-package-structure--import-strategy)
13. [Key Design Decisions & Trade-offs](#13-key-design-decisions--trade-offs)

---

## 1. Overall Design Philosophy

CyberStat follows a **tool-augmented LLM agent** pattern (also called ReAct):

```
User Question → LLM Reasons → LLM Calls Tools → Tools Return Data → LLM Responds
```

The key architectural principle is **separation of concerns across four boundaries**:

| Boundary           | Responsibility                          | Files                          |
|--------------------|-----------------------------------------|--------------------------------|
| **UI Layer**       | Layout, rendering, session state        | `app.py`, `components/`        |
| **Agent Layer**    | LLM orchestration, tool wrappers       | `agent/analyst.py`             |
| **Logic Layer**    | Pure data functions (stats, ML, viz)   | `agent/tools/`                 |
| **Data Layer**     | Typed models, state contracts          | `agent/models/`                |

The agent layer **never** imports Streamlit. The logic layer **never** knows about the LLM.
The UI layer **never** does statistics. Each can change independently.

---

## 2. The Streamlit Execution Model & Why It Matters

Streamlit has an unusual execution model that drives many design choices:

```
User clicks something
       ↓
Streamlit re-runs app.py from line 1 to the end
       ↓
Every widget re-renders
       ↓
Page appears in browser
```

**This means:**
- Every variable is re-created on every interaction (no persistent local variables)
- `st.session_state` is the ONLY way to persist data between reruns
- Code order matters — widgets render top-to-bottom
- `st.rerun()` explicitly triggers another full re-execution

This is why:
- `init_session_state()` uses `if "key" not in st.session_state` guards (idempotent init)
- The welcome message check (`if not st.session_state.messages`) is placed before the layout
- After processing a chat message or upload, `st.rerun()` is called to refresh the whole page

---

## 3. Session State Architecture

Five keys, three roles:

```
st.session_state
│
├── DATA (what we're analyzing)
│   ├── df: pd.DataFrame          ← the uploaded CSV/Excel data
│   └── current_file: str         ← "filename_size" to detect re-uploads
│
├── AGENT MEMORY (conversation continuity)
│   ├── messages: list[ChatMessage]     ← human-readable chat bubbles
│   └── pydantic_messages: list         ← PydanticAI's internal format (includes tool calls)
│
└── ANALYSIS STATE (what the agent has discovered)
    ├── analysis_state: AnalysisState   ← findings, recommendations, activity log, data summary
    └── plots: list[(bytes, str)]       ← visualization images + descriptions
```

**Design choice: Two separate message lists.**

`messages` holds simple `ChatMessage(role, content)` objects for the UI. It only has user
and assistant messages — clean and display-ready.

`pydantic_messages` holds PydanticAI's internal format, which includes system messages, tool
calls, tool results, and assistant responses. This is what the LLM needs for context. The two
lists serve different audiences (human vs. LLM) and are maintained independently.

**Design choice: Mutable state passed by reference.**

`analysis_state` is a Pydantic model stored in session state. It's passed into
`AnalysisDependencies` by reference. When a tool calls `ctx.deps.state.add_finding(...)`,
it mutates the same object. No need to "return" the updated state — it's already in
`st.session_state`.

`plots` works differently — a fresh `new_plots = []` is created each request, passed in,
mutated by tools, then `.extend()`-ed into `st.session_state.plots` afterward. This is because
we want to track which plots are "new this turn" vs. "all plots ever."

---

## 4. The Async Bridge: nest_asyncio + asyncio.run()

```python
import nest_asyncio
nest_asyncio.apply()
```

**The problem:** PydanticAI's `agent.run()` is `async` (it makes HTTP calls to Gemini).
Streamlit runs its own event loop internally. Calling `asyncio.run()` inside an already-running
event loop raises `RuntimeError: This event loop is already running`.

**The solution:** `nest_asyncio.apply()` patches Python's asyncio to allow nested event loops.
After this call, `asyncio.run()` works even when called from inside Streamlit's existing loop.

**Where it's called:**

```python
# app.py line 24 — applied once at import time
nest_asyncio.apply()

# app.py line 412 — used when handling chat input
response = asyncio.run(get_agent_response(user_input))
```

**Design choice:** The `get_agent_response()` function is `async def` so it can `await`
PydanticAI. But the Streamlit callback context is synchronous, so `asyncio.run()` bridges
the gap. This is the standard pattern for using async libraries in Streamlit.

---

## 5. PydanticAI — Every Pattern In Detail

### 5a. Agent Construction

```python
# agent/analyst.py lines 147-151
analyst_agent = Agent(
    model="google-gla:gemini-3-pro-preview",
    system_prompt=SYSTEM_PROMPT,
    deps_type=AnalysisDependencies,
)
```

**What each parameter does:**

- `model="google-gla:gemini-3-pro-preview"` — The model string follows PydanticAI's provider
  format: `provider:model_name`. `google-gla` = Google's Generative Language API. The model
  is resolved at runtime via the `GEMINI_API_KEY` environment variable.

- `system_prompt=SYSTEM_PROMPT` — A static string sent with every LLM call. It defines the
  agent's persona, rules, and tool usage guidelines. PydanticAI also supports a dynamic
  alternative (decorator-based) that can read from `ctx.deps` — not used here because the
  prompt doesn't need runtime data.

- `deps_type=AnalysisDependencies` — Tells PydanticAI what type to expect for `deps` in
  `agent.run()` and what type `RunContext[T]` will resolve to in tools. This enables type
  checking at authoring time and runtime validation.

**Design choice: Module-level singleton.**

The agent is created once when `analyst.py` is imported, not per-request. This is correct
because `Agent` is stateless — all mutable state lives in `deps`. Creating it once avoids
re-parsing the system prompt and re-registering tools on every Streamlit rerun.

### 5b. The Dependencies Dataclass

```python
@dataclass
class AnalysisDependencies:
    df: Optional[pd.DataFrame] = None
    state: Optional[AnalysisState] = None
    plots: Optional[list] = None
```

**Why `@dataclass` and not `BaseModel` (Pydantic)?**

- Dependencies hold mutable runtime objects (DataFrames, lists). Pydantic models are designed
  for validation and serialization — they'd try to validate/copy the DataFrame, which is
  expensive and unnecessary.
- `@dataclass` is a simple container with zero overhead. It gives you `__init__`, type hints,
  and IDE autocomplete without any validation or copying.
- PydanticAI's `deps_type` accepts any class — dataclass, Pydantic model, or plain class.

**Why all fields are Optional with defaults?**

The agent can run even without data (to tell the user "please upload a file"). Making
everything optional means you can construct `AnalysisDependencies()` with no arguments for
the initial state, and gradually fill in fields as data arrives.

**Relationship to session state:**

```
                    AnalysisDependencies (per-request, temporary)
                    ┌───────────────────────────────┐
session_state.df ──►│ df = same DataFrame object     │
                    │                               │
session_state      ─►│ state = same AnalysisState    │◄── tools mutate this in place
 .analysis_state    │                               │
                    │ plots = NEW empty list         │◄── tools append to this
                    └───────────────────────────────┘
                                    │
                         discarded after agent.run() returns
```

### 5c. RunContext — The Injection Mechanism

```python
def describe_data(ctx: RunContext[AnalysisDependencies]) -> str:
    ctx.deps.df        # → the DataFrame
    ctx.deps.state     # → the AnalysisState
    ctx.deps.plots     # → the plots list
```

**What happens internally:**

1. You call `agent.run(msg, deps=my_deps)`
2. PydanticAI stores `my_deps` internally
3. When the LLM requests a tool call, PydanticAI constructs a `RunContext` wrapping `my_deps`
4. It calls your tool function with this `RunContext` as the first argument
5. Your function accesses `ctx.deps` to reach the shared state

**Critical behavior: `ctx` is invisible to the LLM.**

When PydanticAI generates the JSON schema for a tool to send to Gemini, it **skips** any
parameter named `ctx` with a `RunContext` type. The LLM has no idea it exists. This is how
PydanticAI separates "parameters the LLM provides" from "context the framework injects."

### 5d. Tool Registration — The @agent.tool Decorator

```python
@analyst_agent.tool
def some_tool(ctx: RunContext[AnalysisDependencies], param1: str, param2: int = 5) -> str:
    """
    Description the LLM reads to decide when to call this tool.
    
    Args:
        param1: Description of param1 (the LLM reads this too)
        param2: Description of param2 with default behavior
    """
    return "result string that goes back to the LLM"
```

**The decorator does three things at import time:**

1. **Registers** the function in `analyst_agent`'s internal tool registry
2. **Parses** the function signature to build a JSON schema (skipping `ctx`)
3. **Extracts** the docstring as the tool description

**At runtime, for each tool call:**

1. PydanticAI deserializes the LLM's JSON arguments into Python values
2. Constructs `RunContext` with your deps
3. Calls the function: `some_tool(ctx=run_context, param1="value", param2=3)`
4. Takes the return string and sends it back to the LLM

### 5e. Tool Signatures — What the LLM Sees vs. What It Doesn't

For this function:

```python
@analyst_agent.tool
def compare_groups(
    ctx: RunContext[AnalysisDependencies],  # HIDDEN from LLM
    numeric_column: str,                     # VISIBLE: {"type": "string"}
    group_column: str                        # VISIBLE: {"type": "string"}
) -> str:
```

**The LLM receives a tool definition like:**

```json
{
  "name": "compare_groups",
  "description": "Compare a numeric variable across two groups using a t-test...",
  "parameters": {
    "type": "object",
    "properties": {
      "numeric_column": {
        "type": "string",
        "description": "The numeric column to compare (e.g., 'salary', 'score')"
      },
      "group_column": {
        "type": "string",
        "description": "The categorical column that defines the groups..."
      }
    },
    "required": ["numeric_column", "group_column"]
  }
}
```

**Design choices in parameter types:**

- `str` is used even for lists (like `columns` in PCA/K-means) because LLMs are better at
  producing comma-separated strings than JSON arrays. The wrapper parses them:
  ```python
  col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
  ```

- Empty string `""` is used as the default for optional string parameters (like `color_column`
  in scatter), not `None`, because PydanticAI maps `None` defaults to optional JSON schema
  fields differently. The wrapper converts `""` to `None`:
  ```python
  color = color_column if color_column else None
  ```

- `int` with a sentinel value (like `n_components: int = 0` meaning "all") is preferred over
  `Optional[int]` for the same reason — simpler for the LLM.

### 5f. agent.run() — The Orchestration Loop

```python
result = await analyst_agent.run(
    user_message,
    deps=deps,
    message_history=message_history
)
```

**This single call orchestrates an entire conversation turn:**

```
                    ┌──────────────────────────────────────┐
                    │         PydanticAI agent.run()        │
                    │                                      │
Step 1:             │  Build prompt:                       │
                    │    system_prompt + message_history    │
                    │    + user_message + tool definitions  │
                    │                                      │
Step 2:             │  Send to Gemini API ──────────────►  │
                    │                                      │
Step 3:             │  ◄──────────── Gemini responds       │
                    │                                      │
                    │  Response is text-only? → DONE (go   │
                    │  to Step 6)                          │
                    │                                      │
Step 4:             │  Response has tool_calls? →          │
                    │    For each tool_call:               │
                    │      a) Deserialize arguments        │
                    │      b) Create RunContext(deps)      │
                    │      c) Call the Python function     │
                    │      d) Collect the return string    │
                    │                                      │
Step 5:             │  Send tool results back to Gemini    │
                    │  → Go to Step 3 (loop)              │
                    │                                      │
Step 6:             │  Return RunResult:                   │
                    │    .output = final text              │
                    │    .all_messages() = full history    │
                    └──────────────────────────────────────┘
```

**The loop can repeat multiple times.** In a single user turn, the LLM might:
1. Call `list_columns()` to see what's available
2. Call `get_correlations()` to find relationships
3. Call `create_scatter()` to visualize the strongest one
4. Call `record_finding()` to save the insight
5. Produce a final text response summarizing everything

Each of steps 1-4 is a round-trip to Gemini. PydanticAI handles this loop automatically.

### 5g. RunResult — Extracting Outputs

```python
return result.output, result.all_messages(), new_plots, deps.df
```

- `result.output` → `str` — The LLM's final text response (the chat bubble content)
- `result.all_messages()` → `list` — Every message from this and all prior turns:
  system prompt, user messages, tool calls, tool results, assistant responses

### 5h. Message History — Multi-Turn Memory

```
Turn 1: agent.run("What columns are there?", history=None)
  → LLM calls list_columns()
  → LLM responds: "Your dataset has 8 columns: age, salary..."
  → result.all_messages() = [system, user1, tool_call_1, tool_result_1, assistant1]
  → stored in st.session_state.pydantic_messages

Turn 2: agent.run("Are age and salary correlated?", history=[system, user1, ...])
  → LLM already knows the columns from Turn 1's history
  → LLM calls get_correlations() + create_scatter()
  → result.all_messages() = [system, user1, ..., user2, tool_calls_2, assistant2]
  → stored (replaces previous)

Turn 3: agent.run("What about income?", history=[system, user1, ..., user2, ...])
  → LLM has full context of both prior turns
  → ...
```

**Design choice: Full history replacement, not append.**

`result.all_messages()` returns the COMPLETE history including all prior turns. So
`st.session_state.pydantic_messages` is always replaced wholesale, not appended to. This
simplifies the logic — no risk of double-counting messages.

---

## 6. System Prompt Engineering

The system prompt (lines 50-142 of `analyst.py`) is structured in deliberate sections:

### Section 1: Persona & Capabilities (lines 50-58)
Establishes the agent's role. Short and direct. The LLM uses this to understand WHAT it is.

### Section 2: Behavioral Rules (lines 60-65)
A numbered sequence of HOW to behave. The ordering matters — "understand what they want"
comes before "use tools." This guides the LLM's reasoning sequence.

### Section 3: Data Verification Rules (lines 67-88)
The most heavily emphasized section. Uses bold, caps, and WRONG/RIGHT examples.

**Design choice: Explicit anti-patterns.**

Instead of just saying "use tools first," the prompt shows a concrete WRONG example
("The groups appear to have similar values") and a RIGHT example with specific numbers.
LLMs respond much better to contrastive examples than abstract rules.

### Section 4: Tool Usage Guide (lines 90-106)
Maps tool names to use cases. This is critical because the LLM has 20+ tools — without
guidance it might use the wrong one.

### Section 5: ML Pipeline (lines 108-131)
A numbered step-by-step pipeline with "DON'T SKIP" warnings.

**Design choice: Prescriptive pipeline.**

ML tools have dependencies (PCA must run before PCA plot, K-means before cluster plot). Rather
than relying on the LLM to figure this out, the prompt explicitly prescribes the order. The
tool return strings also reinforce this ("NEXT STEP: Call create_pca_plot()").

### Section 6: Communication Style (lines 133-136)
Short style guidelines.

### Section 7: Hard Constraints (lines 138-142)
Final rules that should always apply.

---

## 7. Tool Construction Patterns

### 7a. The Three-Layer Tool Pattern

Every tool follows a consistent three-layer structure:

```
Layer 1: @agent.tool WRAPPER (in analyst.py)
    ├── Receives RunContext
    ├── Validates prerequisites (data loaded? plots available?)
    ├── Logs activity to state
    ├── Calls the pure function
    ├── Formats return string for the LLM
    └── Handles errors with try/except

Layer 2: PURE FUNCTION (in agent/tools/*.py)
    ├── Receives raw arguments (DataFrame, column names, etc.)
    ├── Performs the computation
    ├── Returns structured data (string, or tuple of df+stats, or tuple of bytes+str+stats)
    └── Raises exceptions on failure

Layer 3: LIBRARY CODE (pandas, scipy, sklearn, matplotlib)
    └── The actual computation
```

**Why three layers?**

- The wrapper knows about PydanticAI (`RunContext`, `deps`) but not about pandas internals
- The pure function knows about pandas/scipy/sklearn but not about PydanticAI
- This means you can test pure functions without mocking the agent
- You can swap the LLM framework without touching the statistics code

### 7b. Statistical Tool Functions

**Shared pattern:** `def function(df: pd.DataFrame, ...) -> str`

All statistical tools:
1. Take a DataFrame and column names as input
2. Validate that columns exist (`if column not in df.columns: return error`)
3. Compute statistics using pandas/scipy
4. Return a formatted string

**Example: `perform_ttest` (agent/tools/ttest.py)**

```python
def perform_ttest(df, column, group_column) -> str:
    # 1. Validate inputs
    if column not in df.columns: return error
    if group_column not in df.columns: return error
    groups = df[group_column].dropna().unique()
    if len(groups) != 2: return error

    # 2. Split data by group
    group1_data = df[df[group_column] == groups[0]][column].dropna()
    group2_data = df[df[group_column] == groups[1]][column].dropna()

    # 3. Compute statistics
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

    # 4. Interpret results
    significance = "significant" if p_value < 0.05 else "not significant"
    effect_interp = "small" if abs(cohens_d) < 0.5 else ...

    # 5. Return formatted string
    return f"T-Test Results: ...\nt-statistic: {t_stat}\np-value: {p_value}\n..."
```

**Design choice: Error strings, not exceptions.**

Statistical tools return error messages as strings rather than raising exceptions. This is
deliberate — the error string goes back to the LLM, which can then explain the problem to
the user in natural language. If they raised exceptions, the wrapper would need to catch them
and the LLM would only see a generic "Error" message.

**Design choice: Interpretation included.**

Every statistical tool includes human-readable interpretation alongside raw numbers
(e.g., "small effect", "highly significant", "not normally distributed"). This helps the
LLM explain results without having to know statistical thresholds.

### 7c. Visualization Tool Functions

**Shared pattern:** `def function(df, ...) -> tuple[bytes, str, dict]`

All visualization tools return a consistent 3-tuple:

| Element | Type    | Purpose                                      |
|---------|---------|----------------------------------------------|
| `[0]`   | `bytes` | PNG image data (rendered by `st.image()`)    |
| `[1]`   | `str`   | Description text (shown in expander title)   |
| `[2]`   | `dict`  | Statistics extracted during plotting          |

**Example: `create_histogram` structure:**

```python
def create_histogram(df, column, bins=30) -> tuple[bytes, str, dict]:
    # 1. Validate
    if column not in df.columns: raise ValueError(...)

    # 2. Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # 3. Plot data with theme colors
    ax.hist(data, bins=bins, color=THEME['neon_blue'])
    ax.axvline(data.mean(), color=THEME['neon_pink'], linestyle='--')

    # 4. Style axes
    ax.set_xlabel(format_label(column))
    ax.set_title(format_label(column))

    # 5. Save to bytes (BytesIO pipeline)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)

    # 6. Calculate statistics
    stats = {
        "count": len(data),
        "mean": round(float(data.mean()), 2),
        "median": ..., "std": ..., "skewness": ...,
    }

    # 7. Return the triple
    return buf.getvalue(), f"Histogram showing distribution of {column}", stats
```

**Design choice: Viz functions raise, stat functions return error strings.**

Visualization functions raise `ValueError` on bad input (e.g., column not found). The
`@agent.tool` wrapper in `analyst.py` catches these with `try/except` and converts them
to error strings for the LLM. This is because viz functions are more complex (they create
matplotlib figures that need cleanup) so exception-based flow is cleaner.

### 7d. ML Tool Functions (PCA & K-Means)

**Shared pattern:** `def function(df, ...) -> tuple[pd.DataFrame, dict]`

ML tools return a modified DataFrame (with new columns added) plus a statistics dictionary.

**The PCA pipeline in detail:**

```python
def perform_pca(df, columns=None, n_components=None):
    # 1. Select numeric columns (or user-specified subset)
    numeric_df = df[columns].select_dtypes(include=[np.number])

    # 2. Drop rows with missing values
    clean_df = numeric_df.dropna()

    # 3. Standardize (StandardScaler: mean=0, std=1)
    #    Required because PCA is sensitive to feature scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df)

    # 4. Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled_data)

    # 5. Add PC columns to ORIGINAL dataframe (not the cleaned subset)
    #    NaN for rows that were dropped during cleaning
    result_df = df.copy()
    for i, col in enumerate(pc_columns):
        result_df[col] = np.nan
        result_df.loc[clean_df.index, col] = transformed[:, i]

    # 6. Build statistics (variance explained, top features, etc.)
    return result_df, stats
```

**Design choice: Copy-and-augment, not replace.**

`result_df = df.copy()` ensures the original columns are preserved. PCA doesn't remove
existing data — it adds new PC1, PC2, ... columns alongside it. This means after PCA,
the user can still ask about original columns AND use PC columns for clustering.

**Design choice: NaN for dropped rows.**

Rows with missing values can't go through PCA (StandardScaler requires complete data).
Instead of dropping them entirely, PCA fills their PC columns with NaN. This keeps the
DataFrame aligned with the original row indices.

**The DataFrame mutation chain:**

```
Original df (uploaded)
    │
    ▼  run_pca() adds PC1, PC2, PC3...
df + PC columns
    │
    ▼  run_kmeans() adds 'cluster'
df + PC columns + cluster column
    │
    ▼  create_cluster_plot() reads PC1, PC2, cluster
Visualization rendered
```

Each step flows through `ctx.deps.df`:
```python
# In run_pca wrapper:
ctx.deps.df = result_df      # Now deps.df has PC columns

# In run_kmeans wrapper:
ctx.deps.df = result_df      # Now deps.df has PC + cluster columns
```

Back in `app.py`, `deps.df` is returned and stored:
```python
if updated_df is not None:
    st.session_state.df = updated_df
```

### 7e. State-Writing Tools (Findings & Recommendations)

These tools have NO analytical logic. They exist solely as a bridge from the LLM to the UI:

```
LLM decides something is important
    ↓
Calls record_finding(title="...", description="...", importance="high")
    ↓
Wrapper calls ctx.deps.state.add_finding(...)
    ↓
AnalysisState.findings list grows
    ↓
On next st.rerun(), analysis_panel.py reads state.findings and renders them
```

**Design choice: String-based enums.**

`importance` and `priority` are `str`, not Python enums. This is because the LLM generates
these values from the docstring description ("low", "medium", or "high"). PydanticAI would
need extra work to validate enums, and the LLM occasionally produces variations (e.g.,
"HIGH" vs "high"). Using plain strings with a rendering-side fallback is more forgiving.

**Design choice: Comma-separated string for lists.**

`record_finding` takes `related_columns: str = ""` (comma-separated) instead of `list[str]`.
This is the same pattern as PCA/K-means columns — LLMs produce better comma-separated strings
than JSON arrays. The wrapper splits it: 
```python
columns = [c.strip() for c in related_columns.split(",") if c.strip()]
```

---

## 8. The Visualization Paradigm

### 8a. The Return Signature Convention

Every visualization function returns `tuple[bytes, str, dict]`. This convention is enforced
by consistency, not by an interface or base class. All callers can rely on:

```python
image_bytes, description, stats = tools.create_<anything>(...)
```

The `@agent.tool` wrapper then does two things with this triple:
1. **Appends `(image_bytes, description)` to `ctx.deps.plots`** → goes to UI
2. **Formats `stats` as a string and returns it** → goes to LLM

This separation is the core of the visualization paradigm: **the user sees the image, the
LLM reads the numbers.**

### 8b. The Theme System

All plots share a single theme defined in `agent/tools/visualizations/__init__.py`:

```python
# Colors dictionary exported as THEME
CYBER_COLORS = {
    'background': '#0d1117',
    'plot_bg': '#161b22',
    'text': '#e6edf3',
    'neon_blue': '#00d4ff',
    'neon_pink': '#ff0080',
    # ...
}
THEME = CYBER_COLORS

# 8-color palette for data series
CYBER_PALETTE = ['#00d4ff', '#ff0080', '#00ff9f', '#b026ff', ...]

# Applied globally via matplotlib rcParams
plt.style.use('dark_background')
plt.rcParams.update({ ... })

# Also applied to seaborn
sns.set_palette(CYBER_PALETTE)
sns.set_style("darkgrid", { ... })
```

**Design choice: Global rcParams, not per-plot styling.**

By setting `plt.rcParams` at module import time, every plot automatically inherits the theme.
Individual plot functions only need to override specific elements (like the histogram's mean
line color). This is the "convention over configuration" approach — most styling is invisible
to the plot functions.

**Design choice: `matplotlib.use('Agg')` at the top.**

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

This MUST be called before any other matplotlib import. The 'Agg' backend renders to memory
buffers (BytesIO) without needing a display. Essential for server-side rendering in Streamlit.

### 8c. The format_label() Utility

```python
format_label("daily_steps")      → "Daily Steps"
format_label("customer_clv")     → "Customer CLV"
format_label("dailySteps")       → "Daily Steps"
format_label("col_purchase_amt") → "Purchase Amount"
```

**Pipeline:**
1. Strip common prefixes (`col_`, `feat_`, `feature_`, `var_`, `field_`)
2. Split camelCase (`dailySteps` → `daily Steps`)
3. Replace `_` and `-` with spaces
4. For each word:
   - If it's a known abbreviation → use the mapping (`id` → `ID`, `clv` → `CLV`)
   - If it's a small word and not first → lowercase (`of`, `the`, `in`)
   - Otherwise → capitalize

**Design choice: Function, not a column rename.**

Labels are computed on-the-fly when rendering plots, not stored. This means the DataFrame
keeps its original column names (important for tool calls — the LLM needs `daily_steps`,
not `Daily Steps`) while the plots show clean labels.

### 8d. The BytesIO Pipeline

Every visualization follows this exact sequence:

```python
# 1. Create figure
fig, ax = plt.subplots(figsize=(5, 3.5))

# 2. Plot data...

# 3. Save to BytesIO buffer (in-memory PNG)
buf = io.BytesIO()
fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
buf.seek(0)

# 4. Close figure (free memory — critical for server-side rendering)
plt.close(fig)

# 5. Return raw bytes
return buf.getvalue(), description, stats
```

**Why BytesIO?** Streamlit's `st.image()` can accept raw bytes. No temp files needed.

**Why `plt.close(fig)`?** Matplotlib keeps references to all figures. In a server that
handles many requests, not closing figures causes memory leaks.

**Why `bbox_inches='tight'`?** Removes whitespace padding around the plot, making it fit
better in the grid layout.

### 8e. Stats-Alongside-Image Pattern

The most distinctive design pattern in the visualization system:

```python
# In the histogram wrapper (analyst.py):
image_bytes, description, stats = tools.create_histogram(ctx.deps.df, column, bins)
ctx.deps.plots.append((image_bytes, description))  # Image → user's screen

# Stats → LLM's context
return f"""Histogram created for {column}...
DISTRIBUTION STATISTICS - Use these in your response:
- Mean: {stats['mean']}
- Median: {stats['median']}
- Skewness: {stats['skewness']} ({stats['skew_interpretation']})"""
```

**The LLM cannot see images.** It's a text model. But it needs accurate numbers to describe
the chart. So every visualization tool computes the same statistics that the chart displays
and returns them as text. The prompt reinforces this:

```
WRONG: "The distribution appears roughly normal" (guessing from nothing)
RIGHT: "With a skewness of 0.12, the distribution is approximately symmetric" (from stats)
```

---

## 9. Pydantic Models — The Data Contracts

### Model Hierarchy

```
AnalysisState (BaseModel)
├── data_summary: Optional[DataSummary]
│   ├── filename: str
│   ├── rows: int
│   ├── columns: int
│   ├── column_names: list[str]
│   ├── numeric_columns: list[str]
│   ├── categorical_columns: list[str]
│   └── missing_values: dict[str, int]
│
├── findings: list[Finding]
│   ├── title: str
│   ├── description: str
│   ├── importance: str              ← "low" | "medium" | "high"
│   └── related_columns: list[str]
│
├── recommendations: list[Recommendation]
│   ├── title: str
│   ├── description: str
│   └── priority: str                ← "low" | "medium" | "high"
│
└── activity_log: list[ActivityLogEntry]
    ├── timestamp: datetime           ← auto-set via default_factory=datetime.now
    ├── action: str
    └── details: Optional[str]
```

Separate from the analysis hierarchy:

```
ChatMessage (BaseModel)
├── role: str                         ← "user" | "assistant"
├── content: str
└── timestamp: datetime
```

### Why Pydantic BaseModel (not dataclass) for models?

- **Validation at construction:** If something tries to create `Finding(title=123)`, Pydantic
  raises a validation error. The agent is an LLM — type safety at the boundary matters.
- **Serialization:** `model.model_dump()` / `model.model_dump_json()` work out of the box
  if you ever want to save/load sessions.
- **Field descriptions:** `Field(description="...")` documents the field purpose and can be
  used for schema generation.

### Why dataclass for AnalysisDependencies?

Dependencies hold mutable runtime objects (DataFrames). Pydantic would try to validate them,
which is expensive and unnecessary. Dataclass is a zero-cost container.

**Rule of thumb in this codebase:**
- Data that **crosses boundaries** (user → state → UI) = Pydantic BaseModel
- Data that **stays inside the agent runtime** = dataclass or plain Python

### Helper Methods on AnalysisState

```python
class AnalysisState(BaseModel):
    def add_activity(self, action, details=None): ...
    def add_finding(self, title, description, importance, related_columns): ...
    def add_recommendation(self, title, description, priority): ...
```

These are convenience methods that construct the child model and append it. They keep the
tool wrapper code clean:

```python
# Without helpers:
ctx.deps.state.activity_log.append(ActivityLogEntry(action="...", details="..."))

# With helpers:
ctx.deps.state.add_activity("...", "...")
```

---

## 10. The File Upload Pipeline

```python
# app.py — right column
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if "current_file" not in st.session_state or st.session_state.current_file != file_key:
        df = process_uploaded_file(uploaded_file)
        if df is not None:
            # 1. Store DataFrame
            st.session_state.df = df
            st.session_state.current_file = file_key

            # 2. Create summary model
            summary = create_data_summary(df, uploaded_file.name)
            st.session_state.analysis_state.data_summary = summary

            # 3. Log activity
            st.session_state.analysis_state.add_activity(...)

            # 4. Add welcome message
            st.session_state.messages.append(ChatMessage(...))

            # 5. Trigger re-render
            st.rerun()
```

**Design choice: `file_key` for deduplication.**

Streamlit re-runs the entire script on every interaction. Without the `file_key` check, the
same file would be "uploaded" on every rerun. The key (`name_size`) fingerprints the file
so it's only processed once.

**Design choice: Summary created in app.py, not in the agent.**

`create_data_summary()` runs in app.py because it needs to happen immediately (before any
agent call). The summary populates the side panel right away. The agent can then reference
this summary through `ctx.deps.state.data_summary` if needed.

---

## 11. The Visualization Grid Renderer

```python
# app.py — visualization pane
plots_reversed = list(reversed(st.session_state.plots))

for row_start in range(0, len(plots_reversed), 2):
    col1, col2 = st.columns(2)

    with col1:
        img_bytes, description = plots_reversed[row_start]
        with st.expander(f"📊 {description[:40]}...", expanded=True):
            st.image(img_bytes, use_container_width=True)

    if row_start + 1 < len(plots_reversed):
        with col2:
            img_bytes, description = plots_reversed[row_start + 1]
            with st.expander(f"📊 {description[:40]}...", expanded=True):
                st.image(img_bytes, use_container_width=True)
```

**How the grid works:**

- Plots are reversed (newest first)
- Iterated in steps of 2 (`range(0, n, 2)`)
- Each pair creates a 2-column row
- Odd-numbered last plot gets a half-width row (second column left empty)
- Each plot is wrapped in an expander with the description as the title
- `expanded=True` means all plots are visible by default
- Descriptions are truncated to 40 chars to prevent the expander title from wrapping

---

## 12. Package Structure & Import Strategy

### The re-export pattern

Each package uses `__init__.py` to re-export its contents, creating a flat import surface:

```python
# agent/tools/__init__.py
from .describe import describe_dataframe
from .column_info import get_column_info
from .correlation import calculate_correlation
# ... etc

# Usage in analyst.py:
from . import tools
tools.describe_dataframe(df)    # Clean namespace
```

**Design choice: `from . import tools` (module import) not `from .tools import *`.**

In `analyst.py`, tools are accessed as `tools.describe_dataframe(...)` rather than imported
directly. This makes it clear in every wrapper function that the computation is delegated
to the tools package. It also avoids name collisions (the wrapper `describe_data` vs. the
pure function `describe_dataframe`).

### The visualization sub-package

```
agent/tools/visualizations/
├── __init__.py         ← Theme config + format_label() + re-exports
├── histogram.py        ← create_histogram()
├── scatter.py          ← create_scatter_plot()
├── heatmap.py          ← create_correlation_heatmap()
├── boxplot.py          ← create_box_plot()
├── bar_chart.py        ← create_bar_chart()
├── pca_plot.py         ← create_pca_scatter(), create_scree_plot()
└── cluster_plot.py     ← create_cluster_scatter(), create_elbow_plot(), ...
```

The `__init__.py` does double duty:
1. Configures matplotlib/seaborn theme (must happen before any plot function imports)
2. Re-exports all visualization functions

Individual plot files import from their parent: `from . import THEME, CYBER_PALETTE, format_label`

---

## 13. Key Design Decisions & Trade-offs

### Decision 1: Strings as tool return types

**Choice:** All tools return `str`, never structured data.
**Why:** PydanticAI sends tool results back to the LLM as text. The LLM needs to read and
interpret the results. Formatted strings with clear section headers ("CORRELATION STATISTICS",
"GROUP STATISTICS") are easier for the LLM to parse than JSON.
**Trade-off:** Less programmatic — you can't easily extract values from the return. But the
only consumer is the LLM, which prefers natural text.

### Decision 2: Stats returned alongside images

**Choice:** Visualization tools return statistics as a dict, formatted to a string by the wrapper.
**Why:** The LLM can't see images. It needs the numbers to describe the chart accurately.
**Trade-off:** Some duplication (statistics computed during plotting AND formatted as text).
But this ensures the LLM's description matches what the user sees.

### Decision 3: Mutable deps.df for ML tools

**Choice:** PCA and K-means modify `ctx.deps.df` (adding columns).
**Why:** Subsequent tools (create_pca_plot, create_cluster_plot) need the new columns.
**Trade-off:** Side-effect-heavy. The DataFrame's column set changes during a single turn.
But it enables the natural pipeline: PCA → plot → cluster → plot.

### Decision 4: Module-level agent, per-request deps

**Choice:** `analyst_agent` is created once; `AnalysisDependencies` is created per request.
**Why:** The agent configuration (model, system prompt, tools) doesn't change between requests.
Only the data changes. This is the standard PydanticAI pattern.
**Trade-off:** None — this is the intended usage.

### Decision 5: Two message histories

**Choice:** `messages` (for UI) and `pydantic_messages` (for LLM) are separate.
**Why:** The UI only shows user/assistant text. The LLM needs tool calls and results too.
Mixing them would either clutter the UI or starve the LLM of context.
**Trade-off:** Slight complexity in keeping both in sync.

### Decision 6: Cyberpunk theme as global rcParams

**Choice:** Theme applied once at import time via `plt.rcParams.update()`.
**Why:** Every plot automatically inherits the theme. No per-plot boilerplate.
**Trade-off:** Global state. If another library changes rcParams, the theme could break.
But in a single-purpose app like this, it's fine.

### Decision 7: No base class for tools

**Choice:** Tool functions are standalone functions, not methods on a class.
**Why:** PydanticAI's `@agent.tool` decorator works with plain functions. Adding a class
hierarchy would add complexity without benefit.
**Trade-off:** The `tuple[bytes, str, dict]` return convention is enforced by consistency,
not by a type system. A Protocol or ABC could enforce this, but the codebase is small enough
that convention works.

### Decision 8: Empty string defaults instead of Optional for LLM params

**Choice:** `color_column: str = ""` instead of `color_column: Optional[str] = None`
**Why:** LLMs generate better JSON when optional fields have simple string defaults rather
than null values. The wrapper converts `""` → `None`.
**Trade-off:** Extra conversion code in wrappers, but more reliable LLM behavior.

---

*End of detailed architecture overview.*
