from app.rag.context_builder import ContextBuilder


def test_build_context_formats_multiple_chunks_in_input_order() -> None:
    candidates = [
        {
            "context": "second ranked chunk",
            "source": "orders.md",
            "track": "hybrid",
            "index": 4,
            "metadata": {"rerank_score": 0.8},
        },
        {
            "context": "third ranked chunk",
            "source": "shipping.md",
            "track": "hybrid",
            "index": 2,
            "metadata": {"rerank_score": 0.6},
        },
    ]

    result = ContextBuilder().build_context(candidates)

    assert result == {
        "context": "second ranked chunk\n\nthird ranked chunk",
        "sources": [
            {"source": "orders.md", "index": 4},
            {"source": "shipping.md", "index": 2},
        ],
    }
    assert "hybrid" not in result["context"]
    assert "rerank_score" not in result["context"]
    assert "metadata" not in result["context"]


def test_build_context_supports_existing_content_and_chunk_index_fields() -> None:
    result = ContextBuilder().build(
        [
            {
                "content": "existing chunk",
                "source": "policy.md",
                "chunk_index": 3,
                "rrf_score": 0.02,
            }
        ]
    )

    assert result == {
        "context": "existing chunk",
        "sources": [{"source": "policy.md", "index": 3}],
    }
    assert "rrf_score" not in result["context"]


def test_build_context_handles_empty_or_missing_candidates() -> None:
    builder = ContextBuilder()

    empty_result = {"context": "", "sources": []}
    assert builder.build_context(None) == empty_result
    assert builder.build_context([]) == empty_result
    assert builder.build_context([None, {}, {"source": "missing-content.md"}]) == empty_result


def test_build_context_skips_empty_content_and_handles_missing_fields() -> None:
    result = ContextBuilder().build_context(
        [
            {"context": "   ", "source": "empty.md"},
            {"content": "usable chunk"},
        ]
    )

    assert result == {
        "context": "usable chunk",
        "sources": [{"source": "", "index": None}],
    }
