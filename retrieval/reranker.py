from __future__ import annotations

import asyncio
import logging

from sentence_transformers import CrossEncoder

from models import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-scores (query, passage) pairs with a cross-encoder. It's far more
    accurate than the bi-encoder retrieval but too slow to run over everything,
    so we only apply it to the fused shortlist.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        logger.info("Loading reranker model '%s' on %s ...", model_name, device)
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks], show_progress_bar=False)
        rescored = [c.model_copy(update={"score": float(s)}) for c, s in zip(chunks, scores)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_n]

    async def arerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.rerank, query, chunks, top_n)
