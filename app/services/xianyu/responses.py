"""Response builders shared by the Xianyu item handlers.

These helpers deliberately return the existing API dictionaries.  Keeping that
shape here lets the handlers focus on evidence decisions without changing the
public ``/chat`` contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from app.services.chat_response import non_rag_response


BUYER_HANDOFF_REPLY = "稍等我看看"
_HUMAN_REVIEW_LANGUAGE = re.compile(
    r"(?:需要|让|请|等|等待).{0,8}(?:卖家|人工).{0,8}(?:确认|处理)"
    r"|(?:我|帮你).{0,8}(?:问|联系).{0,8}卖家"
    r"|(?:转|交给).{0,8}(?:卖家|人工).{0,8}(?:处理|确认)",
    re.IGNORECASE,
)


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


def requires_human_handoff(text: str) -> bool:
    """Reject buyer-facing model text that asks them to wait for seller review."""

    normalized = str(text or "").strip().rstrip("。！？!?").strip()
    if normalized.startswith(("缺少依据", "资料不足", "无法确认")):
        return True
    return bool(_HUMAN_REVIEW_LANGUAGE.search(normalized))


def merge_partial_response(
    partial: Mapping[str, object],
    knowledge: Mapping[str, object],
) -> dict[str, object]:
    """Escalate an unresolved combined turn without leaking a partial buyer reply."""

    merged = dict(partial)
    if partial.get("can_answer") is False or knowledge.get("can_answer") is False:
        merged["answer"] = BUYER_HANDOFF_REPLY
        merged["action"] = "handoff"
        merged["next_step"] = "human_handoff"
        merged["can_answer"] = False
        merged["reason"] = str(
            partial.get("reason") or knowledge.get("reason") or "combined_answer_unavailable"
        )
        merged["sources"] = knowledge.get("sources", partial.get("sources", []))
        merged["results"] = knowledge.get("results", partial.get("results", []))
        merged["reliability"] = knowledge.get("reliability", partial.get("reliability"))
        return merged
    merged["answer"] = join_answers(
        [
            str(knowledge.get("answer") or ""),
            str(partial.get("answer") or ""),
        ]
    )
    merged["sources"] = knowledge.get("sources", [])
    merged["results"] = knowledge.get("results", [])
    merged["reliability"] = knowledge.get("reliability")
    merged["can_answer"] = False
    return merged


def reply(
    query: str,
    answer: str,
    item: Mapping[str, object],
) -> dict[str, object]:
    """Build a deterministic reply based solely on confirmed item facts."""

    if requires_human_handoff(answer):
        return handoff(query, "generated_reply_requires_human_review", item)
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
    """Escalate a missing item/question context instead of guessing."""

    response = common_handoff(query, "buyer_question_requires_clarification")
    if item_id is not None:
        response["item_id"] = item_id
    return response


def item_conflict(
    query: str,
    answer: str,
    *,
    item_id: str | None = None,
) -> dict[str, object]:
    """Escalate conflicting item identity without selecting one by guesswork."""

    del answer  # Kept for compatibility with existing callers.
    response = common_handoff(query, "item_context_conflict")
    if item_id is not None:
        response["item_id"] = item_id
    return response


def handoff(
    query: str,
    answer: str,
    item: Mapping[str, object],
    prepared: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the fixed buyer handoff while retaining an internal reason."""

    response = non_rag_response(
        query,
        BUYER_HANDOFF_REPLY,
        can_answer=False,
        route="xianyu",
        action="handoff",
    )
    response.update(
        {
            "reason": answer,
            "item_id": item["item_id"],
            "item_info": dict(item),
            "next_step": "human_handoff",
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
    """Create a fixed buyer handoff when no concrete item is available."""

    response = non_rag_response(
        query,
        BUYER_HANDOFF_REPLY,
        can_answer=False,
        route="xianyu",
        action="handoff",
    )
    response["reason"] = reason
    response["next_step"] = "human_handoff"
    return response
