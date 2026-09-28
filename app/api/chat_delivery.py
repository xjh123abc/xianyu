"""Authenticated internal delivery receipts for generated price proposals."""

from __future__ import annotations

import secrets
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.chat import chat_service
from config.settings import settings


router = APIRouter()
DeliveryState = Literal["LOCAL_SUBMITTED", "CONFIRMED", "FAILED", "UNKNOWN"]


class DeliveryRequest(BaseModel):
    """A channel receipt identifies a proposal but can never set its amount."""

    model_config = ConfigDict(extra="forbid")

    chat_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    delivery_state: DeliveryState


@router.post("/internal/chat/delivery")
async def delivery_receipt(
    request: DeliveryRequest,
    internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, object]:
    """Apply one delivery transition after authenticating the channel caller."""

    configured_token = str(getattr(settings, "xianyu_delivery_token", ""))
    if not configured_token or not internal_token or not secrets.compare_digest(
        configured_token, internal_token
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="internal delivery authentication failed",
        )
    return await chat_service.record_delivery(
        request.chat_id,
        turn_id=request.turn_id,
        proposal_id=request.proposal_id,
        delivery_state=request.delivery_state,
    )
