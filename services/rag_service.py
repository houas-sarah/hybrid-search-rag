from __future__ import annotations

import logging

from config import Settings
from generation import BaseGenerator, build_generator
from ingestion import IngestionPipeline
from models import IngestResponse, QueryResponse, RetrievedChunk, SourceCitation
from retrieval import (
    CrossEncoderReranker,
    Embedder,
    HybridRetriever,
    SparseIndex,
    VectorStore,
)

logger = logging.getLogger(__name__)


class RAGService:
    """Holds the shared resources (models, indexes, LLM client) and exposes the
    two things the API needs: ingest() and query(). Everything expensive is
    loaded once in create().
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        reranker: CrossEncoderReranker,
        generator: BaseGenerator,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._generator = generator
        self._pipeline = IngestionPipeline(settings, embedder, vector_store, sparse_index)
        self._retriever = HybridRetriever(settings, embedder, vector_store, sparse_index, reranker)

    @classmethod
    def create(cls, settings: Settings) -> "RAGService":
        settings.ensure_directories()

        embedder = Embedder(settings.embedding_model, settings.device)
        vector_store = VectorStore(settings.chroma_persist_dir, settings.collection_name)
        sparse_index = SparseIndex()
        sparse_index.load(settings.sparse_index_path)  # reuse a previous run if present
        reranker = CrossEncoderReranker(settings.reranker_model, settings.device)
        generator = build_generator(settings)

        logger.info("RAGService ready (embedding dim=%d).", embedder.dimension)
        return cls(settings, embedder, vector_store, sparse_index, reranker, generator)

    async def ingest(self) -> IngestResponse:
        num_docs, num_chunks = await self._pipeline.run()
        return IngestResponse(
            documents_indexed=num_docs,
            chunks_indexed=num_chunks,
            docs_dir=str(self._settings.docs_dir),
        )

    async def retrieve(self, query: str, top_n: int | None = None) -> list[RetrievedChunk]:
        """Retrieval only (no generation) — used by the evaluation harness."""
        return await self._retriever.retrieve(query, top_n=top_n)

    async def query(self, query: str, top_n: int | None = None) -> QueryResponse:
        chunks = await self._retriever.retrieve(query, top_n=top_n)
        answer = await self._generator.generate(query, chunks)
        sources = [
            SourceCitation(source=c.source, text=c.text, score=round(c.score, 4))
            for c in chunks
        ]
        return QueryResponse(query=query, answer=answer, sources=sources)

    @property
    def is_indexed(self) -> bool:
        return self._vector_store.count() > 0 or self._sparse_index.is_ready
