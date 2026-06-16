"""Tests for the two-tier LLM client."""

from unittest.mock import patch, MagicMock

import pytest

from quantmind.llm_client import LLMClient, Provider, ProviderUnavailableError


class TestProviderDetection:
    """M2 — _detect_provider logic tests."""

    @patch("quantmind.llm_client.httpx.get")
    def test_ollama_reachable(self, mock_get):
        """Ollama responds 200 → Provider.OLLAMA."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = LLMClient()
        assert client.provider == Provider.OLLAMA
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags", timeout=3
        )

    @patch("quantmind.llm_client.httpx.get")
    def test_ollama_timeout_fallback_openrouter(self, mock_get):
        """Ollama times out, OPENROUTER_API_KEY set → Provider.OPENROUTER."""
        mock_get.side_effect = TimeoutError("Ollama unreachable")

        client = LLMClient()
        assert client.provider == Provider.OPENROUTER
        assert "openrouter" in client.active_host

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.OPENROUTER_API_KEY", "")
    def test_both_unavailable(self, mock_get):
        """Ollama down + no OPENROUTER_API_KEY → Provider.UNAVAILABLE."""
        mock_get.side_effect = ConnectionError("No Ollama")

        client = LLMClient()
        assert client.provider == Provider.UNAVAILABLE
        assert client.active_host == ""

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.OPENROUTER_API_KEY", "")
    def test_complete_raises_when_unavailable(self, mock_get):
        """Calling complete() on UNAVAILABLE → ProviderUnavailableError."""
        mock_get.side_effect = ConnectionError("No Ollama")

        client = LLMClient()
        with pytest.raises(ProviderUnavailableError):
            client.complete(system="", user="test")

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.OPENROUTER_API_KEY", "")
    def test_embed_raises_when_unavailable(self, mock_get):
        """Calling embed() on UNAVAILABLE → tries sentence-transformers fallback."""
        mock_get.side_effect = ConnectionError("No Ollama")

        client = LLMClient()
        # This should not raise — it falls back to sentence-transformers
        # But sentence-transformers may not be installed in CI
        try:
            result = client.embed("test text")
            assert isinstance(result, list)
            assert len(result) > 0
        except ModuleNotFoundError:
            pytest.skip("sentence-transformers not installed")


class TestOllamaEmbed:
    """_ollama_embed integration."""

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.LLMClient._ollama_embed")
    def test_ollama_embed_called_when_provider_is_ollama(
        self, mock_embed, mock_get
    ):
        """embed() delegates to _ollama_embed when provider is Ollama."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        mock_embed.return_value = [0.1] * 768

        client = LLMClient()
        result = client.embed("some text")

        mock_embed.assert_called_once_with("some text")
        assert len(result) == 768

    @patch("quantmind.llm_client.httpx.get")
    def test_ollama_embed_failure_falls_back(self, mock_get):
        """If _ollama_embed raises, embed() falls back to sentence-transformers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.object(LLMClient, "_ollama_embed", side_effect=RuntimeError("Ollama error")):
            client = LLMClient()
            try:
                result = client.embed("test")
                assert isinstance(result, list)
            except ModuleNotFoundError:
                pytest.skip("sentence-transformers not installed")


class TestOllamaComplete:
    """_ollama_complete integration (mocked SDK)."""

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.LLMClient._ollama_complete")
    def test_ollama_complete_called(self, mock_complete, mock_get):
        """complete() delegates to _ollama_complete when provider is Ollama."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        mock_complete.return_value = "Test response"

        client = LLMClient()
        result = client.complete(
            system="You are a test assistant.",
            user="Say hello.",
        )

        mock_complete.assert_called_once_with(
            "You are a test assistant.", "Say hello."
        )
        assert result == "Test response"

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.LLMClient._ollama_complete_stream")
    def test_ollama_complete_stream_yields_tokens(self, mock_stream, mock_get):
        """complete_stream yields tokens from _ollama_complete_stream."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        mock_stream.return_value = iter(["Hello", " world"])

        client = LLMClient()
        tokens = list(client.complete_stream("system", "user"))
        assert tokens == ["Hello", " world"]


class TestOpenRouterComplete:
    """_openrouter_complete integration (mocked OpenAI SDK)."""

    @patch("quantmind.llm_client.httpx.get")
    @patch("quantmind.llm_client.OPENROUTER_API_KEY", "sk-test")
    def test_openrouter_complete_called(self, mock_get):
        """With Ollama down and OpenRouter key set, _openrouter_complete is called."""
        mock_get.side_effect = TimeoutError("Ollama down")

        with patch("quantmind.llm_client.OpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenRouter response"
            mock_instance.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )

            client = LLMClient()
            result = client.complete(system="s", user="u")
            assert result == "OpenRouter response"


class TestProviderEnumValues:
    """Provider enum string values must match config expectations."""

    def test_provider_values(self):
        assert Provider.OLLAMA.value == "ollama"
        assert Provider.OPENROUTER.value == "openrouter"
        assert Provider.UNAVAILABLE.value == "unavailable"
