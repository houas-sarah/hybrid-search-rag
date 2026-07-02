from __future__ import annotations

import asyncio
import logging

from config import Settings
from models import RetrievedChunk
from retrieval.embedder import Embedder
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse_index import SparseIndex
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense + sparse search in parallel, fused with RRF, then reranked."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        reranker: CrossEncoderReranker,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._reranker = reranker

    async def _dense_search(self, query: str) -> list[RetrievedChunk]:
        embedding = await self._embedder.aembed_query(query)
        return await self._vector_store.asearch(embedding, self._settings.top_k_dense)

    async def _sparse_search(self, query: str) -> list[RetrievedChunk]:
        return await self._sparse_index.asearch(query, self._settings.top_k_sparse)

    async def retrieve(self, query: str, top_n: int | None = None) -> list[RetrievedChunk]:
        top_n = top_n or self._settings.top_n_rerank

        dense, sparse = await asyncio.gather(
            self._dense_search(query),
            self._sparse_search(query),
        )

        fused = reciprocal_rank_fusion([dense, sparse], k=self._settings.rrf_k)
        if not fused:
            return []
        return await self._reranker.arerank(query, fused, top_n)
