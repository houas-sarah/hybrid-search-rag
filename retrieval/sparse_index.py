from __future__ import annotations

import asyncio
import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    """BM25 keyword index. Lives in memory and gets pickled to disk so it
    survives a restart. Holds the same chunks (same ids) as the vector store so
    the two ranked lists can be fused.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._corpus_tokens = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        logger.info("Built BM25 index over %d chunk(s).", len(self._chunks))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
            "corpus_tokens": self._corpus_tokens,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        self._chunks = [Chunk(**c) for c in payload["chunks"]]
        self._corpus_tokens = payload["corpus_tokens"]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        logger.info("Loaded BM25 index (%d chunks) from %s", len(self._chunks), path)
        return self.is_ready

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self.is_ready:
            return []
        assert self._bm25 is not None

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[RetrievedChunk] = []
        for idx in ranked[:k]:
            if scores[idx] <= 0.0:  # skip non-matches
                break
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    metadata=chunk.metadata,
                    score=float(scores[idx]),
                )
            )
        return results

    async def asearch(self, query: str, k: int) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.search, query, k)
