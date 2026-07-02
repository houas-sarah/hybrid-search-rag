"""Integration tests for the ChromaDB dense store (uses a temp persist dir)."""

from models import Chunk
from retrieval.vector_store import VectorStore

CHUNKS = [
    Chunk(id="a.md::0", text="python postgresql redis", source="a.md"),
    Chunk(id="b.md::0", text="configuration port settings", source="b.md"),
]
EMBS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_add_and_nearest_neighbour_search(tmp_path):
    vs = VectorStore(tmp_path / "chroma", "test_docs")
    vs.reset()
    vs.add(CHUNKS, EMBS)

    assert vs.count() == 2
    res = vs.search([0.9, 0.1, 0.0], k=2)
    assert res[0].source == "a.md"
    assert 0.0 <= res[0].score <= 1.0  # cosine distance mapped to similarity


def test_deterministic_ids_upsert_not_duplicate(tmp_path):
    vs = VectorStore(tmp_path / "chroma", "test_docs")
    vs.reset()
    vs.add(CHUNKS, EMBS)
    vs.add(CHUNKS, EMBS)  # same ids again
    assert vs.count() == 2


def test_search_on_empty_collection(tmp_path):
    vs = VectorStore(tmp_path / "chroma", "test_docs")
    vs.reset()
    assert vs.search([1.0, 0.0, 0.0], k=3) == []
