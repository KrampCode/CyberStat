"""Chat message model."""

from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A single message in the chat history.
    """
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="The message text")
    timestamp: datetime = Field(default_factory=datetime.now)
