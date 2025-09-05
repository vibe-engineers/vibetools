import pytest
from unittest.mock import MagicMock, PropertyMock
from vibetools.llms.gemini_wrapper import GeminiWrapper
from vibetools._internal.vibe_config import VibeConfig
from vibetools.exceptions.exceptions import VibeResponseParseException, VibeLlmApiException
from google import genai


@pytest.fixture
def mock_gemini_client():
    return MagicMock(spec=genai.Client)

@pytest.fixture
def config():
    return VibeConfig(system_instruction="You are a test assistant.")


@pytest.fixture
def logger():
    return MagicMock()

@pytest.fixture
def wrapper(mock_gemini_client, config, logger):
    return GeminiWrapper(mock_gemini_client, "test_model", config, logger)

def test_vibe_eval_raw_text(wrapper, mock_gemini_client):
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value="  raw text  ")
    mock_gemini_client.models.generate_content.return_value = mock_response

    result = wrapper.vibe_eval("test prompt")
    assert result == "raw text"
    wrapper.client.models.generate_content.assert_called_once()

def test_vibe_eval_coerced_output(wrapper, mock_gemini_client):
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value='{"key": "value"}')
    mock_gemini_client.models.generate_content.return_value = mock_response

    result = wrapper.vibe_eval("test prompt", return_type=dict)
    assert result == {"key": "value"}

def test_vibe_eval_parse_exception(wrapper, mock_gemini_client):
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value="not a valid dict")
    mock_gemini_client.models.generate_content.return_value = mock_response

    with pytest.raises(VibeResponseParseException):
        wrapper.vibe_eval("test prompt", return_type=dict)

def test_vibe_eval_api_exception(wrapper, mock_gemini_client):
    mock_gemini_client.models.generate_content.side_effect = Exception("API error")

    with pytest.raises(VibeLlmApiException):
        wrapper.vibe_eval("test prompt")


def test_system_instruction_is_passed(wrapper, mock_gemini_client):
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value="raw text")
    mock_gemini_client.models.generate_content.return_value = mock_response

    wrapper.vibe_eval("test prompt")
    wrapper.client.models.generate_content.assert_called_once_with(
        model="test_model",
        contents="test prompt",
        config={"system_instruction": "You are a test assistant."},
    )
