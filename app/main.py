"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router

app = FastAPI(title="E-commerce AI Customer Service")

app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/health", status_code=200)
async def health() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}
