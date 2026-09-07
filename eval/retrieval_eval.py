"""Run the retrieval-only evaluation chain and write per-stage JSONL results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.ingestion.loader import load_configured_knowledge_base
from app.ingestion.pipeline import ensure_knowledge_base_in_sync
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from config.settings import settings


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "eval" / "retrieval_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "results" / "retrieval_results.jsonl"


def _read_cases(path: Path) -> list[dict[str, Any]]:
    """Read every non-empty JSONL record from the evaluation input."""
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation line {line_number} must contain a JSON object")
        if not isinstance(case.get("question"), str):
            raise ValueError(f"Evaluation line {line_number} is missing a string question")
        cases.append(case)
    return cases


def _result_record(result: object, *, score: float | None = None) -> dict[str, Any]:
    """Project an existing retrieval result into the evaluation schema."""
    if isinstance(result, Mapping):
        payload = result
    else:
        payload = getattr(result, "payload", None) or {}

    if not isinstance(payload, Mapping):
        raise ValueError("Retrieval result payload must be a mapping")
    if "content" not in payload or "source" not in payload or "chunk_index" not in payload:
        raise ValueError("Retrieval result is missing content/source/chunk_index")

    raw_score = score
    if raw_score is None:
        raw_score = payload.get("rerank_score", payload.get("score"))
        if raw_score is None:
            raw_score = getattr(result, "score", None)
    if raw_score is None:
        raise ValueError("Retrieval result is missing a score")

    return {
        "source": str(payload["source"]),
        "chunk_index": int(payload["chunk_index"]),
        "score": float(raw_score),
        "content": str(payload["content"]),
    }


def _stage_results(
    results: Sequence[object],
    *,
    score_key: str | None = None,
) -> list[dict[str, Any]]:
    """Add one-based stage ranks and normalize stage scores."""
    normalized: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        if score_key is None:
            normalized.append({"rank": rank, **_result_record(result)})
            continue

        if not isinstance(result, Mapping) or score_key not in result:
            raise ValueError(f"Retrieval result is missing {score_key}")
        normalized.append(
            {"rank": rank, **_result_record(result, score=float(result[score_key]))}
        )
    return normalized


def run_evaluation(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    vector_search: VectorSearch | None = None,
    bm25_search: BM25Search | None = None,
    reranker: Reranker | None = None,
) -> int:
    """Run Dense -> BM25 -> RRF -> Reranker for every input case."""
    if settings is None:
        raise RuntimeError("Project settings are unavailable")

    cases = _read_cases(input_path)
    chunks = load_configured_knowledge_base()
    ensure_knowledge_base_in_sync(chunks)
    dense_search = vector_search or VectorSearch()
    keyword_search = bm25_search or BM25Search(chunks)
    hybrid_search = HybridSearch(
        dense_search,
        keyword_search,
        rrf_k=settings.rrf_k,
    )
    stage_reranker = reranker or Reranker()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for case in cases:
            question = case["question"]

            dense_raw = list(dense_search.search(question, top_k=settings.dense_top_k))
            bm25_raw = list(keyword_search.search(question, top_k=settings.bm25_top_k))
            rrf_raw = hybrid_search.fuse(
                dense_raw,
                bm25_raw,
                top_k=max(settings.dense_top_k, settings.bm25_top_k),
            )
            rerank_raw = stage_reranker.rerank(
                question,
                rrf_raw,
                top_k=settings.reranker_top_k,
            )

            result = dict(case)
            result["dense"] = _stage_results(dense_raw)
            result["bm25"] = _stage_results(bm25_raw)
            result["rrf"] = _stage_results(rrf_raw)
            result["rerank"] = _stage_results(rerank_raw, score_key="rerank_score")
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = run_evaluation(args.input, args.output)
    print(f"Wrote {count} retrieval cases to {args.output}")


if __name__ == "__main__":
    main()
