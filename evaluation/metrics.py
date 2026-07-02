from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_CITATION_RE = re.compile(r"\[([\w./\-]+\.(?:md|markdown|txt))\]")


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def hit_at_k(ranked_sources: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant source is in the top-k, else 0.0."""
    return 1.0 if set(ranked_sources[:k]) & set(relevant) else 0.0


def recall_at_k(ranked_sources: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant sources that appear in the top-k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(ranked_sources[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_sources: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant source (0 if none show up)."""
    relevant = set(relevant)
    for rank, source in enumerate(ranked_sources, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def extract_citations(answer: str) -> list[str]:
    """Filenames the model cited inline, e.g. [readme.md]."""
    return _CITATION_RE.findall(answer)


def citation_faithfulness(answer: str, context_sources: Iterable[str]) -> float | None:
    """Fraction of cited filenames that were actually in the retrieved context.

    1.0 means every citation is grounded; < 1.0 means the model cited a file it
    wasn't given. None if the answer contained no citations.
    """
    cited = set(extract_citations(answer))
    if not cited:
        return None
    return len(cited & set(context_sources)) / len(cited)


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float | None:
    """Fraction of expected keywords present in the answer (a cheap proxy for
    correctness that needs no LLM judge). None if no keywords were given."""
    if not keywords:
        return None
    low = answer.lower()
    return sum(kw.lower() in low for kw in keywords) / len(keywords)
