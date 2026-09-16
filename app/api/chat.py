"""Chat API routes."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    """Request body for a chat question."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    item_id: str | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        """Normalize valid questions and reject whitespace-only input at the API edge."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


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


class OrderResponse(BaseModel):
    """Structured order data available to the customer-facing UI."""

    found: bool
    order_id: str | None = None
    order_status: str | None = None
    logistics_status: str | None = None
    tracking_no: str | None = None


class ItemResponse(BaseModel):
    """Public readonly item facts returned by the Xianyu flow."""

    found: bool
    item_id: str | None = None
    title: str | None = None
    listed_price_cents: int | None = None
    sale_status: str | None = None
    data_source: str | None = None
    updated_at: str | None = None
    facts: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    """Grounded answer and its pipeline decision."""

    query: str
    answer: str | None = None
    reason: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    can_answer: bool | None = None
    next_step: str | None = None
    reliability: ReliabilityResponse | None = None
    results: list[RetrievedResult] = Field(default_factory=list)
    route: str | None = None
    chat_id: str | None = None
    mcp_result: OrderResponse | None = None
    action: str | None = None
    item_id: str | None = None
    item_info: ItemResponse | None = None
    intent: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/chat",
    response_model=Response,
    response_model_exclude_unset=True,
)
async def chat(request: ChatRequest) -> Response:
    """Receive a question and return the grounded pipeline result."""
    service_response = await chat_service.chat_async(
        request.query,
        request.chat_id,
        item_id=request.item_id,
    )
    return Response(**service_response)
