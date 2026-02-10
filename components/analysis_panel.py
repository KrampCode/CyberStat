"""
Analysis Panel Component - The side panel showing structured analysis results.

Streamlit Concepts Demonstrated:
- Components that render based on state
- Expanders for collapsible sections
- Metrics for key numbers
- Markdown for formatted text
"""

import streamlit as st
from agent.models import AnalysisState


def escape_latex(text: str) -> str:
    """
    Escape characters that Streamlit markdown interprets as LaTeX.
    
    The $ symbol triggers LaTeX math mode in Streamlit, so we need to escape it
    when displaying currency values or other text containing $.
    """
    if text is None:
        return ""
    # Escape $ to prevent LaTeX interpretation
    return text.replace("$", r"\$")


def render_analysis_panel(state: AnalysisState) -> None:
    """
    Render the analysis side panel.
    
    This shows:
    1. Data Summary section (when data is loaded)
    2. Key Findings section (populated by agent)
    3. Recommendations section (populated by agent)
    4. Activity Log (collapsible, shows agent actions)
    
    Args:
        state: The current AnalysisState
    """
    # ---------------------------------------------------------------------
    # Data Summary Section
    # ---------------------------------------------------------------------
    st.markdown("#### Data Summary")
    
    if state.data_summary:
        summary = state.data_summary
        
        # Key metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{summary.rows:,}")
        with col2:
            st.metric("Columns", summary.columns)
        
        # Column lists
        if summary.numeric_columns:
            st.markdown(f"**Numeric:** {', '.join(summary.numeric_columns[:5])}"
                       + (f" +{len(summary.numeric_columns)-5} more" if len(summary.numeric_columns) > 5 else ""))
        
        if summary.categorical_columns:
            st.markdown(f"**Categorical:** {', '.join(summary.categorical_columns[:5])}"
                       + (f" +{len(summary.categorical_columns)-5} more" if len(summary.categorical_columns) > 5 else ""))
        
        # Missing values warning
        total_missing = sum(summary.missing_values.values())
        if total_missing > 0:
            st.warning(f"Missing values: {total_missing:,} across {len(summary.missing_values)} columns")
    else:
        st.info("Upload a dataset to see summary statistics")
    
    st.divider()
    
    # ---------------------------------------------------------------------
    # Key Findings Section
    # ---------------------------------------------------------------------
    st.markdown("#### Key Findings")
    
    if state.findings:
        for i, finding in enumerate(state.findings):
            # Label by importance
            if finding.importance == "high":
                label = "[HIGH]"
            elif finding.importance == "medium":
                label = "[MED]"
            else:
                label = "[LOW]"
            
            # Only expand first finding to reduce clutter
            with st.expander(f"{label} {escape_latex(finding.title)}", expanded=(i == 0)):
                st.markdown(escape_latex(finding.description))
                if finding.related_columns:
                    st.caption(f"Related: {', '.join(finding.related_columns)}")
    else:
        st.caption("Findings will appear here as you analyze the data")
    
    st.divider()
    
    # ---------------------------------------------------------------------
    # Recommendations Section
    # ---------------------------------------------------------------------
    st.markdown("#### Recommendations")
    
    if state.recommendations:
        for rec in state.recommendations:
            priority_badge = {
                "high": ":red[HIGH]",
                "medium": ":orange[MEDIUM]",
                "low": ":green[LOW]"
            }.get(rec.priority, "")
            
            st.markdown(f"**{escape_latex(rec.title)}** {priority_badge}")
            st.markdown(escape_latex(rec.description))
            st.markdown("---")
    else:
        st.caption("Recommendations will appear based on analysis")
    
    st.divider()
    
    # ---------------------------------------------------------------------
    # Activity Log (Collapsible)
    # ---------------------------------------------------------------------
    with st.expander("Activity Log", expanded=False):
        if state.activity_log:
            for entry in reversed(state.activity_log):  # Most recent first
                time_str = entry.timestamp.strftime("%H:%M:%S")
                st.markdown(f"**{time_str}** - {escape_latex(entry.action)}")
                if entry.details:
                    st.caption(escape_latex(entry.details))
        else:
            st.caption("No activity yet")


def render_data_preview(df) -> None:
    """
    Render a preview of the loaded dataframe.
    
    Args:
        df: pandas DataFrame to preview
    """
    if df is not None:
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
