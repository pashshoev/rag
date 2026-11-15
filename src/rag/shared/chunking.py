"""Text chunking utilities.

``chunk_size`` / ``overlap`` are measured in **characters** (not tokens).
That keeps behavior easy to reason about for this MVP; token-aware splitting
can be added later as a separate chunker class.
"""

from __future__ import annotations

from rag.shared.types import Chunk, Document

# Prefer natural boundaries before hard-cutting mid-word.
_SEPARATORS = ("\n\n", "\n", " ")


class CharacterChunker:
    """Split a document into overlapping character windows."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        pieces = _split_text(doc.text, self.chunk_size, self.overlap)
        chunks: list[Chunk] = []
        for index, text in enumerate(pieces):
            metadata = {
                **doc.metadata,
                "doc_id": doc.id,
                "chunk_index": index,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "size_unit": "characters",
            }
            chunks.append(
                Chunk(
                    id=f"{doc.id}::{index}",
                    text=text,
                    metadata=metadata,
                )
            )
        return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        hard_end = min(start + chunk_size, length)
        end = hard_end if hard_end >= length else _best_end(text, start, hard_end)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def _best_end(text: str, start: int, hard_end: int) -> int:
    """Move ``hard_end`` left to a separator, but keep enough progress.

    Reject boundaries in the first half of the window so overlap math cannot
    collapse into 1-character steps (which explodes chunk counts).
    """
    window = text[start:hard_end]
    min_idx = len(window) // 2
    for sep in _SEPARATORS:
        idx = window.rfind(sep)
        if idx >= min_idx:
            return start + idx
    return hard_end
