"""Chat API routes."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    """Request body for a chat question."""

    query: str


class Response(BaseModel):
    """Response returned by the chat API."""

    answer: str


@router.post("/chat", response_model=Response)
def chat(request: ChatRequest) -> Response:
    """Receive a question, delegate it to the chat service, and return its response."""
    service_response = chat_service.chat(request.query)
    return Response(answer=service_response)
