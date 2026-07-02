from __future__ import annotations

from collections.abc import Sequence

from models import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge ranked lists using only rank position: score = sum(1 / (k + rank)).

    Because it ignores the raw scores, it happily mixes signals that live on
    different scales (cosine similarity vs. BM25). Chunks are de-duplicated by
    id and the output is sorted by the fused score.
    """
    fused: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(chunk.id, chunk)

    ordered = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [by_id[cid].model_copy(update={"score": fused[cid]}) for cid in ordered]
