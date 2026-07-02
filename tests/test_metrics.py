"""Unit tests for the evaluation metrics (pure functions)."""

from evaluation.metrics import (
    citation_faithfulness,
    dedupe_preserve_order,
    extract_citations,
    hit_at_k,
    keyword_coverage,
    recall_at_k,
    reciprocal_rank,
)


def test_dedupe_preserves_order():
    assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_hit_at_k():
    ranked = ["b.md", "a.md", "c.md"]
    assert hit_at_k(ranked, {"a.md"}, 1) == 0.0   # a is at rank 2
    assert hit_at_k(ranked, {"a.md"}, 2) == 1.0
    assert hit_at_k(ranked, {"z.md"}, 3) == 0.0


def test_recall_at_k():
    ranked = ["a.md", "b.md", "c.md"]
    assert recall_at_k(ranked, {"a.md", "c.md"}, 3) == 1.0
    assert recall_at_k(ranked, {"a.md", "c.md"}, 1) == 0.5
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a.md", "b.md"], {"a.md"}) == 1.0
    assert reciprocal_rank(["a.md", "b.md"], {"b.md"}) == 0.5
    assert reciprocal_rank(["a.md", "b.md"], {"z.md"}) == 0.0


def test_extract_citations():
    answer = "Use Python 3.11 [readme.md] and set the port [configuration.md][readme.md]."
    assert extract_citations(answer) == ["readme.md", "configuration.md", "readme.md"]
    assert extract_citations("no citations here") == []


def test_citation_faithfulness():
    # both citations are in context -> fully grounded
    assert citation_faithfulness("a [x.md] b [y.md]", {"x.md", "y.md"}) == 1.0
    # one cited file wasn't in context -> half grounded
    assert citation_faithfulness("a [x.md] b [z.md]", {"x.md", "y.md"}) == 0.5
    # no citations at all -> undefined
    assert citation_faithfulness("no cites", {"x.md"}) is None


def test_keyword_coverage():
    assert keyword_coverage("It needs Python 3.11 and PostgreSQL", ["3.11", "postgresql"]) == 1.0
    assert keyword_coverage("It needs Python 3.11", ["3.11", "redis"]) == 0.5
    assert keyword_coverage("anything", []) is None
