from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    # id is deterministic ("{source}::{index}") so re-ingesting overwrites
    # instead of duplicating, and the dense/sparse indexes can be joined on it.
    id: str
    text: str
    source: str
    metadata: dict[str, str | int] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    # score is reused across stages: retriever similarity -> RRF score -> rerank score
    score: float = 0.0


class IngestResponse(BaseModel):
    status: str = "ok"
    documents_indexed: int
    chunks_indexed: int
    docs_dir: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)


class SourceCitation(BaseModel):
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]
