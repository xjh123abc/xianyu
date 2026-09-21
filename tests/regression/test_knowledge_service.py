"""S5 checks for the unified evidence-retrieval boundary."""

from __future__ import annotations

import pytest

from app.services.knowledge_service import KnowledgeService


class _RecordingRag:
    def __init__(self) -> None:
        self.warm_up_calls = 0
        self.prepare_calls: list[tuple[str, str | None]] = []

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def prepare(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        self.prepare_calls.append((query, item_id))
        return {"can_answer": True, "context": {"context": "证据"}}


@pytest.mark.parametrize(
    ("scope", "kwargs"),
    [
        ("merchant", {"platform": "xianyu"}),
        ("product", {"product_model": "Canon FTB"}),
        ("platform", {"platform": "xianyu"}),
    ],
)
def test_search_uses_common_evidence_for_non_item_scopes(
    scope: str,
    kwargs: dict[str, str],
) -> None:
    rag = _RecordingRag()
    service = KnowledgeService(lambda: rag)  # type: ignore[arg-type]

    result = service.search("  售后规则  ", scope, **kwargs)  # type: ignore[arg-type]

    assert rag.prepare_calls == [("售后规则", None)]
    assert result["knowledge_scope"] == scope
    assert result["platform"] == kwargs.get("platform")
    assert result["product_model"] == kwargs.get("product_model")


def test_search_uses_item_evidence_for_item_scope() -> None:
    rag = _RecordingRag()
    service = KnowledgeService(lambda: rag)  # type: ignore[arg-type]

    result = service.search("这个修过吗", "item", item_id=" TEST1001 ")

    assert rag.prepare_calls == [("这个修过吗", "TEST1001")]
    assert result["knowledge_scope"] == "item"


def test_search_rejects_invalid_scope_or_missing_item_context() -> None:
    service = KnowledgeService(lambda: _RecordingRag())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="scope"):
        service.search("规则", "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="item_id"):
        service.search("这个修过吗", "item")
    with pytest.raises(ValueError, match="non-empty"):
        service.search("  ", "merchant")


def test_warm_up_delegates_to_the_existing_rag_service() -> None:
    rag = _RecordingRag()
    service = KnowledgeService(lambda: rag)  # type: ignore[arg-type]

    service.warm_up()

    assert rag.warm_up_calls == 1
