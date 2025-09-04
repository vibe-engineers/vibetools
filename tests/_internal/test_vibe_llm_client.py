from unittest.mock import MagicMock, patch

from google import genai
import pytest
from openai import OpenAI

from vibetools._internal.vibe_llm_client import VibeLlmClient
from vibetools.models.exceptions import VibeLlmClientException, VibeTimeoutException
from vibetools.models.vibe_config import VibeConfig


@pytest.fixture
def mock_openai_client():
    return MagicMock(spec=OpenAI)


@pytest.fixture
def mock_gemini_client():
    return MagicMock(spec=genai.Client)


@pytest.fixture
def config():
    return VibeConfig(timeout=100)


@pytest.fixture
def logger():
    return MagicMock()


def test_init_with_openai_client(mock_openai_client, config, logger):
    with patch("vibetools._internal.vibe_llm_client.OpenAiWrapper") as mock_wrapper:
        client = VibeLlmClient(mock_openai_client, "gpt-4", config, logger)
        mock_wrapper.assert_called_once_with(mock_openai_client, "gpt-4", config, logger)
        assert client.llm == mock_wrapper.return_value


def test_init_with_gemini_client(mock_gemini_client, config, logger):
    with patch("vibetools._internal.vibe_llm_client.GeminiWrapper") as mock_wrapper:
        client = VibeLlmClient(mock_gemini_client, "gemini-pro", config, logger)
        mock_wrapper.assert_called_once_with(mock_gemini_client, "gemini-pro", config, logger)
        assert client.llm == mock_wrapper.return_value


def test_init_with_invalid_client(config, logger):
    with pytest.raises(VibeLlmClientException):
        VibeLlmClient(MagicMock(), "test", config, logger)


def test_vibe_eval_timeout(mock_openai_client, config, logger):
    with patch("vibetools._internal.vibe_llm_client.OpenAiWrapper") as mock_wrapper:

        def long_running_eval(*args, **kwargs):
            import time

            time.sleep(0.2)

        mock_wrapper.return_value.vibe_eval.side_effect = long_running_eval
        client = VibeLlmClient(mock_openai_client, "gpt-4", config, logger)
        with pytest.raises(VibeTimeoutException):
            client.vibe_eval("test prompt")


def test_vibe_eval_success(mock_openai_client, config, logger):
    with patch("vibetools._internal.vibe_llm_client.OpenAiWrapper") as mock_wrapper:
        mock_wrapper.return_value.vibe_eval.return_value = "success"
        client = VibeLlmClient(mock_openai_client, "gpt-4", config, logger)
        result = client.vibe_eval("test prompt")
        assert result == "success"
        client.llm.vibe_eval.assert_called_once_with("test prompt", None)
