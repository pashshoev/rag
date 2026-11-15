"""LLM generation — OpenAI-compatible HTTP (Ollama locally, vLLM in prod)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import httpx

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "llama3.2:1b"
DEFAULT_MAX_TOKENS = 200


class Generator(ABC):
    """Abstract chat generator."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model name sent to the server."""

    @abstractmethod
    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return assistant text for a chat message list."""


class OpenAICompatibleGenerator(Generator):
    """Calls ``POST {base_url}/chat/completions`` (Ollama, vLLM, etc.)."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LLM_BASE_URL,
        model_id: str = DEFAULT_LLM_MODEL,
        api_key: str = "ollama",
        timeout_s: float = 120.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        url = f"{self._base_url}/chat/completions"
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_id,
                    "messages": list(messages),
                    "temperature": 0.2,
                    "max_tokens": self._max_tokens,
                },
                timeout=self._timeout_s,
            )
        except httpx.RequestError as exc:
            raise ConnectionError(
                f"Generation server not reachable at {url}: {exc}"
            ) from exc

        if response.status_code == 404:
            raise LookupError(
                f"Model or endpoint not found at {url} "
                f"(model={self._model_id!r}). "
                f"For Ollama: `ollama pull {self._model_id}`."
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Generation server error {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected generation response: {data!r}") from exc


def create_generator(
    *,
    base_url: str = DEFAULT_LLM_BASE_URL,
    model_id: str = DEFAULT_LLM_MODEL,
    api_key: str = "ollama",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Generator:
    return OpenAICompatibleGenerator(
        base_url=base_url,
        model_id=model_id,
        api_key=api_key,
        max_tokens=max_tokens,
    )
