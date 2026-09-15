"""Small response factories shared by chat handlers."""

from __future__ import annotations


def non_rag_response(
    query: str,
    answer: str,
    *,
    can_answer: bool,
    route: str | None = None,
    action: str | None = None,
    item_id: str | None = None,
) -> dict[str, object]:
    """Build the stable response shape used before retrieval results exist."""

    response: dict[str, object] = {
        "query": query,
        "answer": answer,
        "sources": [],
        "results": [],
        "reliability": None,
        "next_step": None,
        "can_answer": can_answer,
    }
    if route is not None:
        response["route"] = route
    if action is not None:
        response["action"] = action
    if item_id is not None:
        response["item_id"] = item_id
    return response
