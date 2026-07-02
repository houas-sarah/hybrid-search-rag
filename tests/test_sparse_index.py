"""Unit tests for the BM25 sparse index and its persistence."""

from models import Chunk
from retrieval.sparse_index import SparseIndex

DOCS = [
    Chunk(id="readme.md::0", text="Install Python 3.11 and PostgreSQL 15", source="readme.md"),
    Chunk(id="config.md::0", text="Set ACME_PORT to change the port number", source="config.md"),
    Chunk(id="tsg.md::0", text="Jobs stuck in pending mean Redis is unreachable", source="tsg.md"),
]


def _index() -> SparseIndex:
    idx = SparseIndex()
    idx.build(DOCS)
    return idx


def test_search_ranks_keyword_match_first():
    res = _index().search("python postgresql", k=3)
    assert res and res[0].source == "readme.md"


def test_no_match_returns_empty():
    assert _index().search("zzzzz nonexistent", k=3) == []


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "bm25.pkl"
    _index().save(path)

    reloaded = SparseIndex()
    assert reloaded.load(path) is True
    assert reloaded.is_ready
    hit = reloaded.search("redis pending", k=1)
    assert hit and hit[0].source == "tsg.md"


def test_load_missing_file_returns_false(tmp_path):
    assert SparseIndex().load(tmp_path / "does-not-exist.pkl") is False


def test_empty_index_is_not_ready():
    assert SparseIndex().is_ready is False
