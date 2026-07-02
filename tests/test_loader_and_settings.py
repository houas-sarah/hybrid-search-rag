"""Unit tests for document loading and configuration."""

import pytest

from config import Settings
from ingestion.loader import load_documents


def test_loads_supported_files_and_skips_others(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nhello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")   # unsupported ext
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")  # blank -> skipped
    sub = tmp_path / "guides"
    sub.mkdir()
    (sub / "setup.md").write_text("nested doc", encoding="utf-8")

    docs = load_documents(tmp_path)
    sources = {d.source for d in docs}

    assert sources == {"a.md", "b.txt", "guides/setup.md"}  # POSIX relative labels
    assert all(d.content for d in docs)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_settings_defaults_match_spec():
    # _env_file=None -> ignore any local .env so we test the code defaults.
    s = Settings(_env_file=None)
    assert s.chunk_size == 512
    assert s.rrf_k == 60
    assert s.top_k_dense == 10 and s.top_k_sparse == 10
    assert s.top_n_rerank == 3


def test_overlap_is_clamped_below_chunk_size():
    s = Settings(_env_file=None, chunk_size=100, chunk_overlap=500)
    assert s.effective_chunk_overlap == 99
