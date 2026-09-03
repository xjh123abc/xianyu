"""Minimal chat service implementation."""


class ChatService:
    """Handle chat business logic outside the FastAPI route."""

    def chat(self, query: str) -> str:
        """Return a minimal response for the received query."""
        return f"已收到你的问题：{query}"
