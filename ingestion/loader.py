from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


@dataclass(slots=True)
class LoadedDocument:
    source: str
    content: str


def load_documents(docs_dir: Path) -> list[LoadedDocument]:
    """Read every .md/.txt file under docs_dir.

    `source` is the path relative to docs_dir (POSIX style) — that's what shows
    up in the citations, so we want it short and stable.
    """
    docs_dir = docs_dir.resolve()
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    documents: list[LoadedDocument] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file: %s", path)
            continue
        if not content:
            continue
        documents.append(LoadedDocument(path.relative_to(docs_dir).as_posix(), content))

    logger.info("Loaded %d document(s) from %s", len(documents), docs_dir)
    return documents
