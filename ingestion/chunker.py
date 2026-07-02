from __future__ import annotations

import re
from collections.abc import Callable

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


class RecursiveCharacterTextSplitter:
    """Split text into overlapping chunks, breaking on paragraph boundaries
    first and falling back to lines, sentences, words and finally characters.

    Length is measured with `length_function`, so the ingestion pipeline can
    pass the embedding model's tokenizer and get token-sized chunks.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        length_function: Callable[[str], int] = len,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length = length_function
        self._separators = separators

    def split_text(self, text: str) -> list[str]:
        return [c for c in (chunk.strip() for chunk in self._split(text, self._separators)) if c]

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        # first separator that actually occurs; "" (per-character) is the fallback
        separator = separators[-1]
        remaining: tuple[str, ...] = ()
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                remaining = separators[i + 1 :]
                break

        splits = self._split_on_separator(text, separator)
        join_sep = "" if separator == "" else separator

        final_chunks: list[str] = []
        pending: list[str] = []
        for piece in splits:
            if self._length(piece) <= self._chunk_size:
                pending.append(piece)
                continue
            # piece too big by itself: flush what we have, then split it further
            if pending:
                final_chunks.extend(self._merge(pending, join_sep))
                pending = []
            if remaining:
                final_chunks.extend(self._split(piece, remaining))
            else:
                final_chunks.append(piece)

        if pending:
            final_chunks.extend(self._merge(pending, join_sep))
        return final_chunks

    @staticmethod
    def _split_on_separator(text: str, separator: str) -> list[str]:
        # the separator is re-inserted on join, so we don't keep it on the pieces
        if separator == "":
            return list(text)
        return [piece for piece in text.split(separator) if piece]

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        # pack pieces up to chunk_size, then slide the window forward keeping
        # chunk_overlap of the tail as the start of the next chunk
        sep_len = self._length(separator)
        chunks: list[str] = []
        window: list[str] = []
        total = 0

        for piece in pieces:
            piece_len = self._length(piece)
            addition = piece_len + (sep_len if window else 0)
            if total + addition > self._chunk_size and window:
                chunks.append(separator.join(window))
                while total > self._chunk_overlap and window:
                    total -= self._length(window[0]) + (sep_len if len(window) > 1 else 0)
                    window.pop(0)
            window.append(piece)
            total += piece_len + (sep_len if len(window) > 1 else 0)

        if window:
            chunks.append(separator.join(window))
        return chunks
