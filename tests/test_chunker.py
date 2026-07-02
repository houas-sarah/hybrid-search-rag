"""Unit tests for the recursive character text splitter."""

import pytest

from ingestion.chunker import RecursiveCharacterTextSplitter

# A word-based length function stands in for the token counter in tests.
word_len = lambda s: len(s.split())


def test_chunks_respect_size_and_cover_all_content():
    text = (
        "Para one sentence one. Para one sentence two.\n\n"
        "Para two has quite a lot of words here to force a split across the window "
        "boundary because it keeps going and going with more and more words.\n\n"
        "Third paragraph is short."
    )
    splitter = RecursiveCharacterTextSplitter(15, 3, length_function=word_len)
    chunks = splitter.split_text(text)

    assert chunks
    assert all(word_len(c) <= 15 for c in chunks)
    # every source word is preserved somewhere
    assert set(text.split()) <= set(" ".join(chunks).split())


def test_consecutive_chunks_overlap():
    words = " ".join(f"w{i}" for i in range(30))
    splitter = RecursiveCharacterTextSplitter(8, 3, length_function=word_len)
    chunks = splitter.split_text(words)

    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        shared = set(a.split()[-4:]) & set(b.split()[:4])
        assert shared, "expected overlap between consecutive chunks"


def test_char_level_fallback_preserves_content():
    big = "x" * 100  # single unbreakable token, no separators
    chunks = RecursiveCharacterTextSplitter(3, 0, length_function=len).split_text(big)
    assert "".join(chunks) == big


def test_single_spacing_is_preserved():
    splitter = RecursiveCharacterTextSplitter(5, 1, length_function=word_len)
    chunks = splitter.split_text("alpha beta gamma delta epsilon zeta")
    assert all("  " not in c for c in chunks), "no doubled spaces on rejoin"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)
