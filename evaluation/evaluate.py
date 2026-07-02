"""Run the evaluation harness over the labeled question set.

    python -m evaluation.evaluate            # retrieval + generation metrics
    python -m evaluation.evaluate --no-gen   # retrieval only (fast, no LLM)

Retrieval metrics (hit@k, recall@k, MRR) score whether the right source docs
come back and how high. Generation metrics check that the answer's citations
are grounded and that expected facts show up.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from config import get_settings
from evaluation.dataset import load_cases
from evaluation.metrics import (
    citation_faithfulness,
    dedupe_preserve_order,
    extract_citations,
    hit_at_k,
    keyword_coverage,
    recall_at_k,
    reciprocal_rank,
)
from services import RAGService

K_VALUES = (1, 3)
RETRIEVE_K = 5


def _mean(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else 0.0


async def run(generate: bool) -> None:
    logging.getLogger().setLevel(logging.WARNING)  # hush model-loading logs

    service = RAGService.create(get_settings())
    if not service.is_indexed:
        print("Index is empty — ingesting first ...")
        await service.ingest()

    cases = load_cases()
    hits = {k: [] for k in K_VALUES}
    recalls = {k: [] for k in K_VALUES}
    rrs: list[float] = []
    faithfulness: list[float | None] = []
    cited_relevant: list[float] = []
    keyword_cov: list[float | None] = []

    print(f"\n{len(cases)} question(s) · generation {'on' if generate else 'off'}\n")

    for case in cases:
        chunks = await service.retrieve(case.question, top_n=RETRIEVE_K)
        ranked = dedupe_preserve_order([c.source for c in chunks])
        relevant = set(case.relevant_sources)

        for k in K_VALUES:
            hits[k].append(hit_at_k(ranked, relevant, k))
            recalls[k].append(recall_at_k(ranked, relevant, k))
        rr = reciprocal_rank(ranked, relevant)
        rrs.append(rr)

        extra = ""
        if generate:
            response = await service.query(case.question)
            context_sources = {s.source for s in response.sources}
            faithfulness.append(citation_faithfulness(response.answer, context_sources))
            cited = set(extract_citations(response.answer))
            cited_relevant.append(1.0 if cited & relevant else 0.0)
            keyword_cov.append(keyword_coverage(response.answer, case.expected_keywords))
            extra = f"  cites={sorted(cited)}"

        mark = "ok " if rr == 1.0 else "   "
        print(f"[{mark}] RR={rr:.2f}  top={ranked}{extra}")
        print(f"        Q: {case.question}")

    print("\nRetrieval")
    for k in K_VALUES:
        print(f"  hit@{k}      {_mean(hits[k]):.2f}")
    for k in K_VALUES:
        print(f"  recall@{k}   {_mean(recalls[k]):.2f}")
    print(f"  MRR         {_mean(rrs):.2f}")

    if generate:
        print("\nGeneration")
        print(f"  citation faithfulness    {_mean(faithfulness):.2f}")
        print(f"  cited a relevant source  {_mean(cited_relevant):.2f}")
        print(f"  keyword coverage         {_mean(keyword_cov):.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--no-gen", action="store_true", help="skip the LLM generation metrics")
    args = parser.parse_args()
    asyncio.run(run(generate=not args.no_gen))


if __name__ == "__main__":
    main()
