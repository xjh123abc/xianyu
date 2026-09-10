"""BM25 keyword retrieval."""

from collections.abc import Callable, Sequence
from collections import Counter
from math import log
import re
from typing import TypedDict

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # Keeps local tests usable before requirements are installed.
    class BM25Okapi:  # type: ignore[no-redef]
        """Small dependency-free BM25Okapi-compatible fallback."""

        def __init__(
            self,
            corpus: Sequence[Sequence[str]],
            k1: float = 1.5,
            b: float = 0.75,
        ) -> None:
            self.k1 = k1
            self.b = b
            self.doc_len = [len(document) for document in corpus]
            self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
            self.doc_freqs = [Counter(document) for document in corpus]
            document_frequency = Counter(
                term for frequencies in self.doc_freqs for term in frequencies
            )
            raw_idf = {
                term: log(len(self.doc_freqs) - frequency + 0.5)
                - log(frequency + 0.5)
                for term, frequency in document_frequency.items()
            }
            average_idf = (
                sum(raw_idf.values()) / len(raw_idf) if raw_idf else 0.0
            )
            self.idf = {
                term: value if value >= 0 else 0.25 * average_idf
                for term, value in raw_idf.items()
            }

        def get_scores(self, query: Sequence[str]) -> list[float]:
            if not self.doc_freqs:
                return []

            scores = [0.0] * len(self.doc_freqs)
            average_length = self.avgdl or 1.0
            for term in query:
                idf = self.idf.get(term, 0.0)
                for index, frequencies in enumerate(self.doc_freqs):
                    frequency = frequencies.get(term, 0)
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * self.doc_len[index] / average_length
                    )
                    if frequency:
                        scores[index] += idf * frequency * (self.k1 + 1) / denominator
            return scores

from app.ingestion.chunker import Chunk


class BM25Result(TypedDict):
    """A chunk returned by BM25 retrieval."""

    content: str
    score: float
    source: str
    chunk_index: int
    scope: str | None
    item_id: str | None


def _tokenize(text: str) -> list[str]:
    """Tokenize Latin words and CJK unigrams/bigrams for mixed text."""
    tokens = []
    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(token)
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


class BM25Search:
    """Search a sequence of chunks using BM25 keyword matching."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = (
            BM25Okapi([_tokenize(chunk.content) for chunk in self.chunks])
            if self.chunks
            else None
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> list[BM25Result]:
        """Return the top matching chunks and their BM25 scores."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        candidate_indexes = [
            index
            for index, chunk in enumerate(self.chunks)
            if filter_fn is None or filter_fn(chunk)
        ]
        ranked_indexes = sorted(
            candidate_indexes,
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        results: list[BM25Result] = []
        for index in ranked_indexes:
            result: BM25Result = {
                "content": self.chunks[index].content,
                "score": float(scores[index]),
                "source": self.chunks[index].source,
                "chunk_index": index,
            }
            if self.chunks[index].scope is not None:
                result["scope"] = self.chunks[index].scope
            if self.chunks[index].item_id is not None:
                result["item_id"] = self.chunks[index].item_id
            results.append(result)
        return results
