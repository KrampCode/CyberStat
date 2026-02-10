"""Chat Interface Component."""

import streamlit as st
from agent.models import ChatMessage


def escape_latex(text: str) -> str:
    """
    Escape characters that Streamlit markdown interprets as LaTeX.
    The $ symbol triggers LaTeX math mode in Streamlit.
    """
    if text is None:
        return ""
    return text.replace("$", r"\$")


def render_chat_history(messages: list[ChatMessage]) -> None:
    """
    Render the chat message history.
    
    Args:
        messages: List of ChatMessage objects
    """
    # Custom avatars matching cyberpunk theme (SVG circles)
    # Cyber pink circle for user
    pink_circle = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%23ff0080'/></svg>"
    # Cyber blue circle for assistant
    blue_circle = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%2300d4ff'/></svg>"
    
    avatars = {
        "user": pink_circle,
        "assistant": blue_circle
    }
    
    for msg in messages:
        avatar = avatars.get(msg.role, None)
        with st.chat_message(msg.role, avatar=avatar):
            st.markdown(escape_latex(msg.content))


def render_chat_input() -> str | None:
    """
    Render the chat input box and return user input if any.
    
    Returns:
        The user's message, or None if no input
    """
    return st.chat_input("Ask about your data...")
