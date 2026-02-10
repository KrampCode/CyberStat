"""
CyberStat - Data Analytics Chat Agent

Main Streamlit application that provides:
- Two-column layout (chat + analysis panel)
- File upload for datasets
- Chat interface connected to PydanticAI agent

Streamlit Concepts Demonstrated:
- Page configuration
- Session state management
- Async execution in Streamlit
- Two-column layouts with bounded containers
- File uploaders
"""

import asyncio
import nest_asyncio
import pandas as pd
import streamlit as st

# Enable nested event loops - required for running async PydanticAI in Streamlit
# Without this, asyncio.run() conflicts with Streamlit's event loop
nest_asyncio.apply()

from agent.analyst import run_analyst, AnalysisDependencies
from agent.models import AnalysisState, DataSummary, ChatMessage
from components.analysis_panel import render_analysis_panel, render_data_preview
from components.chat import render_chat_history


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="CyberStat - Data Analytics Agent",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Cyberpunk theme with neon accents
st.markdown("""
<style>
    /* === CYBERPUNK COLOR PALETTE === */
    :root {
        --neon-blue: #00d4ff;
        --neon-pink: #ff0080;
        --neon-purple: #b026ff;
        --dark-bg: #0d1117;
        --card-bg: #161b22;
        --border-glow: rgba(0, 212, 255, 0.3);
    }
    
    /* === LAYOUT === */
    .block-container {
        padding-top: 1.5rem;
    }
    
    /* === SECTION HEADERS - CYBERPUNK BANNERS === */
    /* Main subheaders (Chat, Analysis, Visualizations) */
    [data-testid="stSubheader"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
        padding: 0.85rem 1.25rem !important;
        border-radius: 10px;
        border: 2px solid var(--neon-blue);
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(0, 212, 255, 0.4),
                    0 0 30px rgba(0, 212, 255, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
    }
    
    /* Glowing top edge highlight */
    [data-testid="stSubheader"]::before {
        content: '';
        position: absolute;
        top: -2px;
        left: 10%;
        right: 10%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--neon-blue), transparent);
        border-radius: 2px;
        box-shadow: 0 0 10px var(--neon-blue);
    }
    
    /* Analysis panel section headers (h4: Data Summary, Key Findings, etc.) */
    h4 {
        background: linear-gradient(135deg, #2d1f3d 0%, #1a1a2e 100%);
        padding: 0.6rem 1rem !important;
        border-radius: 6px;
        border: 1px solid rgba(255, 0, 128, 0.3);
        border-left: 3px solid var(--neon-pink);
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
        font-size: 0.9rem !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3),
                    0 0 15px rgba(255, 0, 128, 0.05);
        letter-spacing: 0.5px;
    }
    
    /* === CONTAINERS WITH SUBTLE GLOW === */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.05), 
                    inset 0 0 30px rgba(0, 0, 0, 0.3);
    }
    
    /* === CHAT INPUT STYLING === */
    .stChatInput {
        border-top: 1px solid rgba(0, 212, 255, 0.3);
        padding-top: 0.75rem;
    }
    
    .stChatInput > div {
        border-radius: 10px !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
    }
    
    .stChatInput input:focus {
        border-color: var(--neon-blue) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        background: rgba(255, 0, 128, 0.05);
        border-radius: 8px;
        padding: 0.6rem 0.75rem !important;
        transition: background 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 0, 128, 0.1);
    }
    
    /* === BUTTONS === */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 128, 0.1) 100%);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        border-color: var(--neon-blue);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        transform: translateY(-1px);
    }
    
    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(255, 0, 128, 0.4);
        border-radius: 10px;
        padding: 0.5rem;
        background: rgba(255, 0, 128, 0.03);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--neon-pink);
        background: rgba(255, 0, 128, 0.06);
    }
    
    /* === METRICS === */
    [data-testid="stMetric"] {
        background: rgba(0, 212, 255, 0.05);
        padding: 0.75rem !important;
        border-radius: 8px;
        border: 1px solid rgba(0, 212, 255, 0.15);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--neon-blue) !important;
    }
    
    /* === DIVIDERS === */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), rgba(255, 0, 128, 0.3), transparent);
        margin: 1.25rem 0 !important;
    }
    
    /* === SPACING === */
    [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.5rem;
    }
    
    /* === CHAT MESSAGES === */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(22, 27, 34, 0.5);
    }
    
    /* User avatar - cyber pink */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #ff0080 0%, #ff4da6 100%) !important;
        border-radius: 8px;
    }
    
    /* Assistant avatar - cyber blue */
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #00d4ff 0%, #00a3cc 100%) !important;
        border-radius: 8px;
    }
    
    /* === DATAFRAME === */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* === MAIN TITLE BANNER === */
    h1 {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
        padding: 1rem 1.5rem !important;
        border-radius: 10px;
        border: 2px solid var(--neon-blue);
        margin-bottom: 0.5rem !important;
        color: #ff9ecb !important;  /* Light cyber pink */
        text-shadow: 0 0 20px rgba(255, 158, 203, 0.5),
                     0 0 40px rgba(255, 158, 203, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(0, 212, 255, 0.3),
                    0 0 40px rgba(0, 212, 255, 0.15);
        letter-spacing: 3px;
    }
    
    /* Caption under title */
    .stCaption {
        padding-left: 0.5rem;
        border-left: 2px solid var(--neon-pink);
        margin-left: 0.5rem;
        opacity: 0.8;
    }
    
    /* === INFO/WARNING BOXES === */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border-left: 3px solid var(--neon-pink);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================
# Streamlit reruns the entire script on every interaction.
# Session state persists data across reruns.

def init_session_state():
    """Initialize session state variables if they don't exist."""
    
    if "messages" not in st.session_state:
        # Chat history
        st.session_state.messages = []
    
    if "analysis_state" not in st.session_state:
        # The structured analysis state shown in the side panel
        st.session_state.analysis_state = AnalysisState()
    
    if "df" not in st.session_state:
        # The loaded dataframe
        st.session_state.df = None
    
    if "pydantic_messages" not in st.session_state:
        # PydanticAI message history for context
        st.session_state.pydantic_messages = None
    
    if "plots" not in st.session_state:
        # List of generated plots: [(image_bytes, description), ...]
        st.session_state.plots = []


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def process_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """
    Process an uploaded file and return a DataFrame.
    
    Supports CSV and Excel files.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def create_data_summary(df: pd.DataFrame, filename: str) -> DataSummary:
    """Create a DataSummary from a DataFrame."""
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    
    return DataSummary(
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        missing_values=missing
    )


async def get_agent_response(user_message: str) -> str:
    """Get a response from the agent and update conversation history."""
    
    # Create a fresh list to collect new plots
    new_plots = []
    
    response, message_history, plots, updated_df = await run_analyst(
        user_message=user_message,
        df=st.session_state.df,
        state=st.session_state.analysis_state,
        message_history=st.session_state.pydantic_messages,
        plots=new_plots
    )
    
    # Store the updated message history for the next call
    # This gives the agent "memory" of the conversation
    st.session_state.pydantic_messages = message_history
    
    # Add any new plots to session state
    if plots:
        st.session_state.plots.extend(plots)
    
    # Update the dataframe if it was modified (e.g., PCA added PC columns, K-means added cluster)
    if updated_df is not None:
        st.session_state.df = updated_df
    
    return response


# =============================================================================
# Initial Welcome Message (before layout to avoid rerun issues)
# =============================================================================
if not st.session_state.messages:
    welcome = """Welcome to CyberStat!

I'm your AI-powered data analytics assistant. Here's how to get started:

1. **Upload a dataset** using the panel on the right (CSV or Excel)
2. **Ask me questions** about your data
3. Watch the **Analysis panel** fill with insights as we explore together

I can help you:
- Understand the structure and quality of your data
- Find correlations and patterns
- Compare groups statistically
- Identify outliers and anomalies

What data would you like to analyze today?"""
    
    st.session_state.messages.append(ChatMessage(role="assistant", content=welcome))


# =============================================================================
# Main Layout
# =============================================================================

# Title
st.title("CyberStat")
st.caption("Your AI-powered data analytics assistant")

# Two-column layout: Chat (main) | Analysis Panel (sidebar)
chat_col, panel_col = st.columns([2, 1], gap="medium")

# -----------------------------------------------------------------------------
# Left Column: Chat Interface  
# -----------------------------------------------------------------------------
with chat_col:
    st.subheader("Chat")
    
    # Create a scrollable container for chat messages (compact to show more viz)
    chat_container = st.container(height=300, border=True)
    
    with chat_container:
        # Render existing messages
        render_chat_history(st.session_state.messages)
    
    # Chat input (outside the scrollable container so it stays fixed)
    user_input = st.chat_input("Ask about your data...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        
        # Get agent response
        with st.spinner("Analyzing..."):
            response = asyncio.run(get_agent_response(user_input))
        
        # Add assistant response to history
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))
        
        # Rerun to update both the chat and analysis panel
        st.rerun()
    
    # -------------------------------------------------------------------------
    # Visualizations Pane (below chat)
    # -------------------------------------------------------------------------
    st.subheader("Visualizations")
    
    if st.session_state.plots:
        # Create a scrollable container for the plot grid
        viz_container = st.container(height=450, border=True)
        
        with viz_container:
            # Show plots in a 2-column grid (newest first)
            plots_reversed = list(reversed(st.session_state.plots))
            
            for row_start in range(0, len(plots_reversed), 2):
                col1, col2 = st.columns(2)
                
                # First plot in row
                with col1:
                    img_bytes, description = plots_reversed[row_start]
                    with st.expander(f"{description[:40]}..." if len(description) > 40 else description, expanded=True):
                        st.image(img_bytes, use_container_width=True)
                
                # Second plot in row (if exists)
                if row_start + 1 < len(plots_reversed):
                    with col2:
                        img_bytes, description = plots_reversed[row_start + 1]
                        with st.expander(f"{description[:40]}..." if len(description) > 40 else description, expanded=True):
                            st.image(img_bytes, use_container_width=True)
        
        # Button to clear plots
        if st.button("Clear Visualizations", type="secondary"):
            st.session_state.plots = []
            st.rerun()
    else:
        st.container(height=100, border=True).markdown(
            "_Visualizations will appear here as the agent analyzes your data._"
        )


# -----------------------------------------------------------------------------
# Right Column: Analysis Panel
# -----------------------------------------------------------------------------
with panel_col:
    st.subheader("Analysis")
    
    # File upload at the top (outside the scrollable container)
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file to analyze"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        # Check if this is a new file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if "current_file" not in st.session_state or st.session_state.current_file != file_key:
            # New file uploaded
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.current_file = file_key
                
                # Create data summary
                summary = create_data_summary(df, uploaded_file.name)
                st.session_state.analysis_state.data_summary = summary
                
                # Log the activity
                st.session_state.analysis_state.add_activity(
                    f"Loaded dataset: {uploaded_file.name}",
                    f"{summary.rows:,} rows, {summary.columns} columns"
                )
                
                # Add a system message about the upload
                welcome_msg = f"I've loaded **{uploaded_file.name}** with {summary.rows:,} rows and {summary.columns} columns. What would you like to know about this data?"
                st.session_state.messages.append(ChatMessage(role="assistant", content=welcome_msg))
                
                st.rerun()
    
    # Show data preview if data is loaded
    if st.session_state.df is not None:
        with st.expander("Data Preview", expanded=False):
            st.dataframe(st.session_state.df.head(50), use_container_width=True)
    
    # Create a scrollable container for analysis results
    # Height set to 550 to roughly align with the chat column's total height
    analysis_container = st.container(height=700, border=True)
    
    with analysis_container:
        render_analysis_panel(st.session_state.analysis_state)
