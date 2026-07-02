"""Unit tests for Reciprocal Rank Fusion."""

from models import RetrievedChunk
from retrieval.fusion import reciprocal_rank_fusion


def _c(cid: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=f"text-{cid}", source="a.md")


def test_rrf_scores_and_ordering():
    dense = [_c("A"), _c("B"), _c("C")]   # ranks 1, 2, 3
    sparse = [_c("B"), _c("D"), _c("A")]  # ranks 1, 2, 3
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    score = {c.id: c.score for c in fused}
    assert score["B"] == 1 / 61 + 1 / 62   # rank2 dense + rank1 sparse
    assert score["A"] == 1 / 61 + 1 / 63   # rank1 dense + rank3 sparse
    # B narrowly beats A; both beat the singletons
    assert [c.id for c in fused][0] == "B"
    assert score["B"] > score["A"] > score["D"]


def test_rrf_dedupes_union_of_ids():
    fused = reciprocal_rank_fusion([[_c("A"), _c("B")], [_c("B"), _c("A")]], k=60)
    assert {c.id for c in fused} == {"A", "B"}
    assert len(fused) == 2


def test_rrf_output_sorted_descending():
    dense = [_c(x) for x in "ABCDE"]
    sparse = [_c(x) for x in "ECABD"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([[], []], k=60) == []
