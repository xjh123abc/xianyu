import pytest

from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService
from config.settings import Settings, settings


def test_retrieval_defaults_are_5_5_60_5() -> None:
    configured = Settings(_env_file=None)

    assert configured.dense_top_k == 5
    assert configured.bm25_top_k == 5
    assert configured.rrf_k == 60
    assert configured.reranker_top_k == 5


def test_retrieval_settings_are_overridable_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DENSE_TOP_K", "8")
    monkeypatch.setenv("BM25_TOP_K", "9")
    monkeypatch.setenv("RRF_K", "70")
    monkeypatch.setenv("RERANKER_TOP_K", "4")

    configured = Settings(_env_file=None)

    assert (configured.dense_top_k, configured.bm25_top_k) == (8, 9)
    assert (configured.rrf_k, configured.reranker_top_k) == (70, 4)


class FakeHybridSearch:
    init_args = None
    search_args = None

    def __init__(self, vector_search, bm25_search, *, rrf_k):
        self.__class__.init_args = (vector_search, bm25_search, rrf_k)

    def search(self, query, *, top_k, dense_top_k, bm25_top_k):
        self.__class__.search_args = (query, top_k, dense_top_k, bm25_top_k)
        return []


class FakeReranker:
    top_k = None

    def rerank(self, query, results, *, top_k):
        self.__class__.top_k = top_k
        return []


class FakePipeline:
    def run(self, query, results):
        return {
            "can_answer": False,
            "next_step": "human_handoff",
            "reliability": {
                "can_answer": False,
                "next_step": "human_handoff",
                "reason": "no_results",
                "top_rerank_score": None,
                "threshold": 0.5,
            },
            "context": None,
            "answer": None,
            "sources": [],
        }


def test_chat_service_uses_configured_retrieval_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dense_top_k", 8)
    monkeypatch.setattr(settings, "bm25_top_k", 9)
    monkeypatch.setattr(settings, "rrf_k", 70)
    monkeypatch.setattr(settings, "reranker_top_k", 4)
    monkeypatch.setattr(chat_service_module, "HybridSearch", FakeHybridSearch)

    service = ChatService(
        vector_search=object(),
        bm25_search=object(),
        reranker=FakeReranker(),
        rag_pipeline=FakePipeline(),
    )
    service.chat("query")

    assert FakeHybridSearch.init_args[2] == 70
    assert FakeHybridSearch.search_args == ("query", 9, 8, 9)
    assert FakeReranker.top_k == 4
