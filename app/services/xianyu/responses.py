"""Response builders shared by the Xianyu item handlers.

These helpers deliberately return the existing API dictionaries.  Keeping that
shape here lets the handlers focus on evidence decisions without changing the
public ``/chat`` contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.services.chat_response import non_rag_response


AUTO_UNAVAILABLE_REPLY = "该问题目前暂无足够信息确认。"
ITEM_CLARIFICATION_REPLY = "请补充商品编号或具体商品信息。"
def item_source(item: Mapping[str, object]) -> dict[str, object]:
    """Represent MCP evidence without pretending it is a document chunk."""

    return {"source": "mcp:get_item_info", "index": item["item_id"]}


def valid_knowledge_sources(
    prepared: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Keep only knowledge references with a source and stable chunk index."""

    raw_sources = prepared.get("sources")
    if not isinstance(raw_sources, list):
        return []
    return [
        source
        for source in raw_sources
        if isinstance(source, Mapping)
        and isinstance(source.get("source"), str)
        and bool(str(source.get("source")).strip())
        and source.get("index") is not None
    ]


def join_answers(parts: Sequence[str], tail: str | None = None) -> str:
    """Join deterministic facts with a grounded knowledge answer."""

    answer_parts = [part.strip() for part in parts if part.strip()]
    if tail is not None and tail.strip():
        answer_parts.append(tail.strip())
    return "\n".join(answer_parts)


def reply(
    query: str,
    answer: str,
    item: Mapping[str, object],
) -> dict[str, object]:
    """Build a deterministic reply based solely on confirmed item facts."""

    return {
        "query": query,
        "route": "xianyu",
        "action": "reply",
        "answer": answer,
        "sources": [item_source(item)],
        "results": [],
        "reliability": None,
        "next_step": None,
        "can_answer": True,
        "item_id": item["item_id"],
        "item_info": dict(item),
    }


def clarification(query: str, item_id: str | None = None) -> dict[str, object]:
    """Ask only for buyer information that can make the next turn answerable."""

    response = non_rag_response(
        query,
        ITEM_CLARIFICATION_REPLY,
        can_answer=False,
        route="xianyu",
        action="clarify",
    )
    response.update(
        {
            "reason": "buyer_question_requires_clarification",
            "next_step": "clarify",
        }
    )
    if item_id is not None:
        response["item_id"] = item_id
    return response


def unavailable(
    query: str,
    answer: str,
    *,
    item: Mapping[str, object] | None = None,
    prepared: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return safe unavailable evidence without requesting human takeover."""

    response = non_rag_response(
        query,
        answer.strip() or "暂无可确认的信息。",
        can_answer=False,
        route="xianyu",
        action="reply",
    )
    if item is not None:
        response.update({"item_id": item["item_id"], "item_info": dict(item)})
    if prepared is not None:
        response.update(
            {
                "sources": valid_knowledge_sources(prepared),
                "results": prepared.get("results", []),
                "reliability": prepared.get("reliability"),
            }
        )
    return response


def item_conflict(
    query: str,
    answer: str,
    *,
    item_id: str | None = None,
) -> dict[str, object]:
    """Ask the buyer to resolve conflicting item identity without guessing."""

    safe_answer = answer.strip() if isinstance(answer, str) else ""
    response = non_rag_response(
        query,
        safe_answer or ITEM_CLARIFICATION_REPLY,
        can_answer=False,
        route="xianyu",
        action="clarify",
    )
    response.update({"reason": "item_context_conflict", "next_step": "clarify"})
    if item_id is not None:
        response["item_id"] = item_id
    return response


def handoff(
    query: str,
    answer: str,
    item: Mapping[str, object],
    prepared: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return an automatic unavailable reply without changing channel ownership.

    The name remains for compatibility with the old expert callers.  It no
    longer means an automatic human handoff.
    """

    response = non_rag_response(
        query,
        AUTO_UNAVAILABLE_REPLY,
        can_answer=False,
        route="xianyu",
        action="reply",
    )
    response.update(
        {
            "reason": answer,
            "item_id": item["item_id"],
            "item_info": dict(item),
            "next_step": None,
            "sources": [item_source(item)],
        }
    )
    if prepared is not None:
        response.update(
            {
                "sources": [item_source(item), *valid_knowledge_sources(prepared)],
                "results": prepared.get("results", []),
                "reliability": prepared.get("reliability"),
            }
        )
    return response


def common_handoff(query: str, reason: str) -> dict[str, object]:
    """Create a safe unavailable reply for an automatic common-knowledge gap.

    Kept as a compatibility name for legacy callers; it never requests human
    takeover and deliberately hides internal failure details from the buyer.
    """

    response = non_rag_response(
        query,
        AUTO_UNAVAILABLE_REPLY,
        can_answer=False,
        route="xianyu",
        action="reply",
    )
    response["reason"] = reason
    response["next_step"] = None
    return response
