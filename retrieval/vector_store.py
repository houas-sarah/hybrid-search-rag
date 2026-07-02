from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent Chroma collection. We feed it our own embeddings so the same
    model is used for indexing and querying, and set cosine space explicitly.
    """

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_or_create()

    def _get_or_create(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass  # nothing to delete on a fresh run
        self._collection = self._get_or_create()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, **c.metadata} for c in chunks],
        )

    def count(self) -> int:
        return self._collection.count()

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        n = self._collection.count()
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunks.append(
                RetrievedChunk(
                    id=cid,
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    metadata={key: val for key, val in meta.items() if key != "source"},
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return chunks

    async def asearch(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.search, query_embedding, k)
