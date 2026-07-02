from __future__ import annotations

from models import RetrievedChunk

SYSTEM_PROMPT = """\
You are a technical-documentation assistant. Answer the question using only the
context passages below. Each passage is prefixed with its source filename in
square brackets, e.g. [readme.md].

Rules:
1. Base every statement on the context. If the answer isn't there, reply exactly:
   "I don't have enough information in the provided documentation to answer that."
2. Put an inline citation after each claim, naming the source filename in square
   brackets, e.g. "The setup requires Python 3.11 [readme.md]."
3. If several passages back a claim, cite them all: [setup.md][config.md].
4. Only cite filenames that appear in the context. Don't invent sources.
5. Be concise and don't go beyond the context.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_context(chunks) if chunks else "(no relevant passages found)"
    return (
        f"Context passages:\n{context}\n\n"
        "----------------------------------------\n"
        f"Question: {query}\n\n"
        "Answer (with inline [filename] citations):"
    )
