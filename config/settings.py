from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"


class Settings(BaseSettings):
    """App config, read from environment variables / a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # paths
    docs_dir: Path = Path("./data/docs")
    chroma_persist_dir: Path = Path("./data/chroma")
    sparse_index_path: Path = Path("./data/bm25/sparse_index.pkl")
    collection_name: str = "internal_docs"

    # chunking, in tokens
    chunk_size: int = Field(512, ge=64)
    chunk_overlap: int = Field(51, ge=0)

    # models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"

    # retrieval
    top_k_dense: int = Field(10, ge=1)
    top_k_sparse: int = Field(10, ge=1)
    rrf_k: int = Field(60, ge=1)
    top_n_rerank: int = Field(3, ge=1)

    # generation
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(1024, ge=1)

    ollama_model: str = "llama3"
    ollama_host: str = "http://localhost:11434"

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_chunk_overlap(self) -> int:
        # guard against someone setting overlap >= size in .env
        return min(self.chunk_overlap, self.chunk_size - 1)

    def ensure_directories(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_index_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
