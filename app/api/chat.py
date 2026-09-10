"""Chat API routes."""

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    """Request body for a chat question."""

    query: str
    scenario: Literal["ecommerce", "xianyu"] = "ecommerce"
    item_id: str | None = None


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


class Response(BaseModel):
    """Grounded answer and its pipeline decision."""

    query: str
    answer: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    can_answer: bool | None = None
    next_step: str | None = None
    reliability: ReliabilityResponse | None = None
    results: list[RetrievedResult] = Field(default_factory=list)
    route: str | None = None
    mcp_result: OrderResponse | None = None
    action: str | None = None
    item_id: str | None = None
    item_info: ItemResponse | None = None


@router.post(
    "/chat",
    response_model=Response,
    response_model_exclude_unset=True,
)
async def chat(request: ChatRequest) -> Response:
    """Receive a question and return the grounded pipeline result."""
    if request.scenario == "ecommerce" and request.item_id is None:
        # Keep the original call shape for existing integrations and tests.
        service_response = await chat_service.chat_async(request.query)
    else:
        service_response = await chat_service.chat_async(
            request.query,
            scenario=request.scenario,
            item_id=request.item_id,
        )
    return Response(**service_response)
