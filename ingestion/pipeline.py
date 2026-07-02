from __future__ import annotations

import asyncio
import logging

from config import Settings
from ingestion.chunker import RecursiveCharacterTextSplitter
from ingestion.loader import load_documents
from models import Chunk
from retrieval.embedder import Embedder
from retrieval.sparse_index import SparseIndex
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Load the docs, chunk them, and (re)build both indexes.

    We rebuild from scratch every time rather than doing incremental updates.
    It's a bit wasteful, but with a docs-sized corpus it's instant and it keeps
    the dense and sparse indexes trivially in sync.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.effective_chunk_overlap,
            length_function=embedder.count_tokens,
        )

    def _chunk_documents(self, documents) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for i, text in enumerate(self._splitter.split_text(doc.content)):
                chunks.append(
                    Chunk(
                        id=f"{doc.source}::{i}",
                        text=text,
                        source=doc.source,
                        metadata={"chunk_index": i},
                    )
                )
        return chunks

    def _run(self) -> tuple[int, int]:
        documents = load_documents(self._settings.docs_dir)
        chunks = self._chunk_documents(documents)

        if not chunks:
            logger.warning("No chunks produced from %s", self._settings.docs_dir)
            self._vector_store.reset()
            self._sparse_index.build([])
            self._sparse_index.save(self._settings.sparse_index_path)
            return 0, 0

        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        self._vector_store.reset()
        self._vector_store.add(chunks, embeddings)

        self._sparse_index.build(chunks)
        self._sparse_index.save(self._settings.sparse_index_path)

        logger.info("Indexed %d document(s), %d chunk(s).", len(documents), len(chunks))
        return len(documents), len(chunks)

    async def run(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._run)
