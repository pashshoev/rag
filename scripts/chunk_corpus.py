#!/usr/bin/env python3
"""Smoke-run chunking on data/corpus and optionally write JSONL for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.shared import CharacterChunker, TextFolderSource

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"
DEFAULT_OUT = REPO_ROOT / "data" / "chunks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help=f"Write chunks to {DEFAULT_OUT} (debug only; not used by the indexer)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = TextFolderSource(args.corpus)
    chunker = CharacterChunker(chunk_size=args.chunk_size, overlap=args.overlap)

    total_docs = 0
    total_chunks = 0
    rows: list[dict] = []

    for doc in source.iter_documents():
        chunks = chunker.split(doc)
        total_docs += 1
        total_chunks += len(chunks)
        print(f"{doc.id}: {len(doc.text)} chars -> {len(chunks)} chunks")
        for chunk in chunks:
            rows.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )

    print(
        f"done — {total_docs} docs, {total_chunks} chunks "
        f"(size={args.chunk_size} chars, overlap={args.overlap})"
    )

    if args.write_jsonl:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
