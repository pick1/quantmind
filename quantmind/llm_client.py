"""Two-tier LLM client — Ollama primary, OpenRouter fallback.

Provider detection runs on instantiation. Embeddings always prefer Ollama
for vector space consistency, falling back to sentence-transformers on CPU
with a prominent warning.
"""

import logging
import os
from enum import Enum

import httpx
from openai import OpenAI

from quantmind.config import (
    OLLAMA_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)

logger = logging.getLogger("quantmind.llm")


class Provider(Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    UNAVAILABLE = "unavailable"


class ProviderUnavailableError(Exception):
    """Raised when no LLM provider is reachable."""


# ═════════════════════════════════════════════════════════════════════════
# Embedding fallback — sentence-transformers on CPU
# ═════════════════════════════════════════════════════════════════════════

_EMBED_FALLBACK_WARNING = (
    "⚠️  Embedding backend switched to sentence-transformers (all-MiniLM-L6-v2) on CPU. "
    "This produces 384-dim vectors, INCOMPATIBLE with the 768-dim ChromaDB collection "
    "created by nomic-embed-text. If you switch back to Ollama later, you MUST rebuild "
    "the ChromaDB collection (delete ./data/chroma and re-ingest)."
)

_sentence_model = None


def _get_sentence_model():
    """Lazy-load the sentence-transformers model (imported here to keep deps optional)."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.warning(_EMBED_FALLBACK_WARNING)
    return _sentence_model


# ═════════════════════════════════════════════════════════════════════════


class LLMClient:
    """Two-tier LLM client with startup provider detection."""

    def __init__(self):
        self.provider, self.active_host = self._detect_provider()
        self._ollama_client = None  # lazy import
        logger.info(
            "LLMClient initialised — provider=%s host=%s",
            self.provider.value,
            self.active_host or "(none)",
        )

    # ── Provider detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_provider() -> tuple[Provider, str]:
        """Health-check Ollama; fall back to OpenRouter or UNAVAILABLE."""
        try:
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            if r.status_code == 200:
                return Provider.OLLAMA, OLLAMA_HOST
        except Exception:
            pass
        if OPENROUTER_API_KEY:
            logger.warning("Ollama unreachable — falling back to OpenRouter")
            return Provider.OPENROUTER, OPENROUTER_BASE_URL
        return Provider.UNAVAILABLE, ""

    # ── Completion ────────────────────────────────────────────────────────

    def complete(self, system: str, user: str) -> str:
        """Synchronous completion via active provider."""
        if self.provider == Provider.OLLAMA:
            return self._ollama_complete(system, user)
        elif self.provider == Provider.OPENROUTER:
            return self._openrouter_complete(system, user)
        raise ProviderUnavailableError(
            "No LLM provider available. Check Ollama host or set OPENROUTER_API_KEY."
        )

    def complete_stream(self, system: str, user: str):
        """Generator yielding content tokens as they arrive."""
        if self.provider == Provider.OLLAMA:
            yield from self._ollama_complete_stream(system, user)
        elif self.provider == Provider.OPENROUTER:
            yield from self._openrouter_complete_stream(system, user)
        else:
            raise ProviderUnavailableError(
                "No LLM provider available. Check Ollama host or set OPENROUTER_API_KEY."
            )

    # ── Embedding ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Embed text. Prefers Ollama; falls back to sentence-transformers on CPU.

        Returns a 768-dim vector when Ollama is available, 384-dim when
        falling back to sentence-transformers. The caller must ensure the
        ChromaDB collection matches.
        """
        if self.provider == Provider.OLLAMA:
            try:
                return self._ollama_embed(text)
            except Exception as exc:
                logger.exception("Ollama embedding failed, trying CPU fallback")
                if isinstance(exc, ProviderUnavailableError):
                    raise
        # Fallback to sentence-transformers
        try:
            logger.warning(_EMBED_FALLBACK_WARNING)
            model = _get_sentence_model()
            return model.encode(text).tolist()
        except Exception:
            logger.exception("Sentence-transformers fallback also failed")
            raise ProviderUnavailableError(
                "Embedding unavailable: Ollama embedding failed and "
                "sentence-transformers fallback is broken (torch CUDA library "
                "has unresolved symbols — install CPU-only torch or fix NCCL). "
                "Fix: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from None

    # ── Ollama implementations ────────────────────────────────────────────

    def _ollama_complete(self, system: str, user: str) -> str:
        from ollama import Client

        client = Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"num_predict": 2048},
        )
        return response["message"]["content"]

    def _ollama_complete_stream(self, system: str, user: str):
        from ollama import Client

        client = Client(host=OLLAMA_HOST)
        stream = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            options={"num_predict": 2048},
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def _ollama_embed(self, text: str) -> list[float]:
        from ollama import Client, ResponseError

        # nomic-embed-text max context is ~8192 tokens. PDF extraction sometimes
        # produces long concatenated strings (bibliographic references with spaces
        # removed), where 1 word can be 80+ chars and tokenize into many tokens.
        # Word-count truncation alone is insufficient — use char-based truncation
        # as a safety net, plus progressive retry on context-length errors.
        MAX_EMBED_CHARS = 12000  # safe: ~3000-6000 tokens for any content type

        if len(text) > MAX_EMBED_CHARS:
            # Trim at character level, splitting at word boundary
            orig_len = len(text)
            truncated = text[:MAX_EMBED_CHARS]
            last_space = truncated.rfind(" ")
            if last_space > MAX_EMBED_CHARS // 2:
                truncated = truncated[:last_space]
            text = truncated
            logger.warning(
                "Truncated embedding input to %d chars (was %d)",
                len(text), orig_len,
            )

        client = Client(host=OLLAMA_HOST)
        try:
            response = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
            return response["embedding"]
        except ResponseError as e:
            error_msg = str(e)
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_msg += " " + e.response.text
            if "context length" in error_msg.lower() and len(text) > 100:
                # Progressive truncation: halve the text and retry
                split = len(text) // 2
                text = text[:split]
                logger.warning(
                    "Context length exceeded, retrying with %d chars",
                    len(text),
                )
                return self._ollama_embed(text)
            raise

    # ── OpenRouter implementations ────────────────────────────────────────

    def _openrouter_complete(self, system: str, user: str) -> str:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    def _openrouter_complete_stream(self, system: str, user: str):
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        stream = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
