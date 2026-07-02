from __future__ import annotations

import asyncio
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """sentence-transformers model used for both embeddings and token counting.

    encode() is blocking and CPU-bound, so the a* helpers push it onto a thread
    to keep the event loop free.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        logger.info("Loading embedding model '%s' on %s ...", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        # get_sentence_embedding_dimension was renamed in sentence-transformers 5
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dimension = get_dim()

    @property
    def dimension(self) -> int:
        return self._dimension

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vector.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
