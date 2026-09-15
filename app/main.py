"""FastAPI application entry point."""

import asyncio
import logging

from fastapi import FastAPI, HTTPException

from app.api.chat import chat_service, router as chat_router
from app.api.conversations import router as conversations_router

app = FastAPI(title="E-commerce AI Customer Service")
logger = logging.getLogger(__name__)

app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/health", status_code=200)
async def health() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}


@app.get("/ready", status_code=200)
async def ready() -> dict[str, str]:
    """Verify that the ordinary RAG chain can load its local dependencies."""

    try:
        await asyncio.to_thread(chat_service.rag_service.warm_up)
        vector_search = getattr(chat_service.rag_service, "vector_search", None)
        verify_collection = getattr(vector_search, "verify_collection", None)
        if not callable(verify_collection):
            raise RuntimeError("RAG vector retriever is unavailable")
        await asyncio.to_thread(verify_collection)
    except Exception:
        logger.exception("RAG readiness check failed")
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "component": "rag"},
        ) from None
    return {"status": "ready", "rag": "ready"}
