"""Data sources that yield Documents. Add new source classes here later."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from rag.shared.types import Document

_HEADER_SEP = "\n---\n"


class DataSource(Protocol):
    def iter_documents(self) -> Iterator[Document]:
        """Yield documents from this source."""


class TextFolderSource:
    """Load ``*.txt`` files from a directory (non-recursive).

    Files written by ``scripts/download_wikipedia.py`` may start with a
    short header (Title/Source/License) separated by ``---``. That header
    is parsed into metadata; only the body is used as ``Document.text``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Corpus folder not found: {self.root}")

    def iter_documents(self) -> Iterator[Document]:
        paths = sorted(self.root.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(f"No .txt files in {self.root}")

        for path in paths:
            raw = path.read_text(encoding="utf-8")
            text, meta = _split_header(raw)
            meta = {
                "source_path": str(path),
                "filename": path.name,
                **meta,
            }
            yield Document(id=path.stem, text=text, metadata=meta)


def _split_header(raw: str) -> tuple[str, dict]:
    """Return (body, metadata) if a ``---`` header is present."""
    if _HEADER_SEP not in raw:
        return raw.strip(), {}

    header, body = raw.split(_HEADER_SEP, maxsplit=1)
    meta: dict = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        key = key.strip().lower()
        value = value.strip()
        if key == "title":
            meta["title"] = value
        elif key == "source":
            meta["source_url"] = value
        elif key == "license":
            meta["license"] = value
    return body.strip(), meta
