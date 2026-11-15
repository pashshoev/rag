#!/usr/bin/env python3
"""Download Wikipedia pages as plain text via the MediaWiki API.

Uses prop=extracts&explaintext=1 so we get readable article body, not HTML.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Five long-form pages — good for chunking demos and on-topic RAG Q&A.
TITLES = [
    "Retrieval-augmented_generation",
    "Vector_database",
    "Word_embedding",
    "Large_language_model",
    "Information_retrieval",
]

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-mvp-corpus-builder/0.1 (portfolio project; local demo)"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


def slugify(title: str) -> str:
    return re.sub(r"[^\w.-]+", "_", title).strip("_").lower()


def fetch_plain_extract(title: str) -> tuple[str, str]:
    """Return (canonical_title, plain_text) for a Wikipedia page title."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": title,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise RuntimeError(f"Wikipedia page not found: {title}")
    extract = (page.get("extract") or "").strip()
    if not extract:
        raise RuntimeError(f"Empty extract for: {title}")
    return page["title"], extract


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    for i, title in enumerate(TITLES):
        canonical, text = fetch_plain_extract(title)
        filename = f"{slugify(title)}.txt"
        path = OUT_DIR / filename
        header = (
            f"Title: {canonical}\n"
            f"Source: https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}\n"
            f"License: CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/\n"
            f"---\n\n"
        )
        path.write_text(header + text + "\n", encoding="utf-8")
        manifest.append(
            {
                "title": canonical,
                "file": filename,
                "url": f"https://en.wikipedia.org/wiki/{title}",
                "chars": str(len(text)),
            }
        )
        print(f"wrote {path.name} ({len(text)} chars) — {canonical}")
        if i < len(TITLES) - 1:
            time.sleep(0.5)  # be polite to the API

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"done — {len(manifest)} pages in {OUT_DIR}")


if __name__ == "__main__":
    main()
