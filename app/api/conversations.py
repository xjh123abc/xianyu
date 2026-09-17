"""Internal conversation-control endpoints for the Xianyu channel."""

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path as FastAPIPath
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.channels.xianyu.control import ChannelControl
from app.channels.xianyu.store import ChannelStore
from config.paths import resolve_project_path
from config.settings import settings


router = APIRouter()


def resolve_channel_database_path() -> Path:
    """Return the configured Xianyu channel database as an absolute path."""

    return resolve_project_path(settings.xianyu_channel_database_path)


def get_channel_store() -> ChannelStore:
    """Create the configured store only when a control request needs it."""

    return ChannelStore(resolve_channel_database_path())


class ResumeAutoRequest(BaseModel):
    """Scope a resume operation to one seller account."""

    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1)

    @field_validator("account_id")
    @classmethod
    def account_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("account_id must not be blank")
        return normalized


class ResumeAutoResponse(BaseModel):
    """The persisted state after resuming one conversation."""

    account_id: str
    chat_id: str
    mode: Literal["AUTO"]
    human_takeover: Literal[False]
    control_version: int


@router.post(
    "/conversations/{chat_id}/resume-auto",
    response_model=ResumeAutoResponse,
)
async def resume_auto(
    chat_id: str = FastAPIPath(min_length=1),
    request: ResumeAutoRequest = ...,
    store: ChannelStore = Depends(get_channel_store),
) -> ResumeAutoResponse:
    """Clear the HUMAN control state for exactly one existing conversation."""

    normalized_chat_id = chat_id.strip()
    if not normalized_chat_id:
        raise HTTPException(status_code=422, detail="chat_id must not be blank")
    state = ChannelControl(store, request.account_id).resume_auto(normalized_chat_id)
    if state is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ResumeAutoResponse(
        account_id=str(state["account_id"]),
        chat_id=str(state["chat_id"]),
        mode="AUTO",
        human_takeover=False,
        control_version=int(state["control_version"]),
    )
