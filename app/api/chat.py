"""Chat API routes."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    """Request body for a chat question."""

    query: str


class RetrievedResult(BaseModel):
    """One reranked chunk kept for retrieval transparency."""

    content: str
    score: float
    source: str
    chunk_index: int


class SourceReference(BaseModel):
    """Source metadata returned alongside the generated answer."""

    source: str
    index: Any


class ReliabilityResponse(BaseModel):
    """Reliability decision exposed to the caller."""

    can_answer: bool
    next_step: str
    reason: str
    top_rerank_score: float | None
    threshold: float


class Response(BaseModel):
    """Grounded answer and its pipeline decision."""

    query: str
    answer: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    can_answer: bool | None = None
    next_step: str | None = None
    reliability: ReliabilityResponse | None = None
    results: list[RetrievedResult] = Field(default_factory=list)
    rag_result: dict[str, Any] | None = None
    mcp_result: dict[str, Any] | None = None


@router.post(
    "/chat",
    response_model=Response,
    response_model_exclude_unset=True,
)
async def chat(request: ChatRequest) -> Response:
    """Receive a question and return the grounded pipeline result."""
    service_response = await chat_service.chat_async(request.query)
    return Response(**service_response)
