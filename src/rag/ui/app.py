"""Streamlit dashboard — calls POST /query/ask."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = float(os.getenv("UI_HTTP_TIMEOUT_S", "180"))


def main() -> None:
    st.set_page_config(page_title="RAG", page_icon="🔎", layout="centered")
    st.title("RAG query")
    st.caption(f"API: `{API_BASE_URL}`")

    query = st.text_area("Query", placeholder="What is retrieval-augmented generation?")
    top_k = st.number_input("top_k", min_value=1, max_value=50, value=3, step=1)

    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Enter a query.")
            return
        with st.spinner("Retrieving + generating…"):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/query/ask",
                    json={"query": query.strip(), "top_k": int(top_k)},
                    timeout=TIMEOUT_S,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")
                if isinstance(exc, httpx.HTTPStatusError):
                    st.code(exc.response.text)
                return

        payload = response.json()
        timings = payload.get("timings") or {}

        st.subheader("Answer")
        st.caption(_format_timings(timings))
        st.text_area(
            "Model response",
            value=str(payload.get("answer") or ""),
            height=160,
            disabled=True,
            label_visibility="collapsed",
        )

        st.subheader("Full response")
        st.json(payload)


def _format_timings(timings: dict) -> str:
    if not timings:
        return "timings: n/a"
    return (
        f"end2end {timings.get('end2end_ms', '—')} ms · "
        f"embedding {timings.get('embedding_ms', '—')} ms · "
        f"retrieval {timings.get('retrieval_ms', '—')} ms · "
        f"generation {timings.get('generation_ms', '—')} ms"
    )


main()
