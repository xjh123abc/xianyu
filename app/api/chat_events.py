"""Authenticated channel events appended to existing chat history."""

from __future__ import annotations

import secrets
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.chat import chat_service
from config.settings import settings


router = APIRouter()


class ConversationEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1, max_length=128)
    chat_id: str = Field(min_length=1, max_length=512)
    event_id: str = Field(min_length=1, max_length=512)
    role: Literal["user", "assistant"]
    source: Literal["buyer_message", "seller_manual"]
    content: str = Field(min_length=1, max_length=10000)

    @field_validator("account_id", "chat_id", "event_id", "content")
    @classmethod
    def trim_nonempty_values(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


@router.post("/internal/chat/events")
async def append_conversation_event(
    request: ConversationEventRequest,
    internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, object]:
    """Append a trusted channel message without invoking chat generation."""
    configured_token = str(getattr(settings, "xianyu_delivery_token", ""))
    if not configured_token or not internal_token or not secrets.compare_digest(
        configured_token, internal_token
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="internal event authentication failed",
        )
    async with chat_service.session_manager.session_lock(request.chat_id):
        appended = chat_service.session_manager.append_external_event(
            request.chat_id,
            f"{request.account_id}:{request.event_id}",
            request.role,
            request.content,
            source=request.source,
        )
    return {"accepted": True, "appended": appended}
