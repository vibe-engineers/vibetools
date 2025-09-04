import pytest
from unittest.mock import MagicMock, PropertyMock
from vibetools.llms.openai_wrapper import OpenAiWrapper
from vibetools.models.vibe_config import VibeConfig
from vibetools.models.exceptions import VibeResponseParseException, VibeLlmApiException

@pytest.fixture
def mock_openai_client():
    return MagicMock()

@pytest.fixture
def config():
    return VibeConfig()

@pytest.fixture
def logger():
    return MagicMock()

@pytest.fixture
def wrapper(mock_openai_client, config, logger):
    return OpenAiWrapper(mock_openai_client, "test_model", config, logger)

def test_vibe_eval_raw_text(wrapper, mock_openai_client):
    mock_response = MagicMock()
    type(mock_response).output_text = PropertyMock(return_value="  raw text  ")
    mock_openai_client.responses.create.return_value = mock_response

    result = wrapper.vibe_eval("test prompt")
    assert result == "raw text"
    wrapper.client.responses.create.assert_called_once()

def test_vibe_eval_coerced_output(wrapper, mock_openai_client):
    mock_response = MagicMock()
    type(mock_response).output_text = PropertyMock(return_value='{"key": "value"}')
    mock_openai_client.responses.create.return_value = mock_response

    result = wrapper.vibe_eval("test prompt", return_type=dict)
    assert result == {"key": "value"}

def test_vibe_eval_parse_exception(wrapper, mock_openai_client):
    mock_response = MagicMock()
    type(mock_response).output_text = PropertyMock(return_value="not a valid dict")
    mock_openai_client.responses.create.return_value = mock_response

    with pytest.raises(VibeResponseParseException):
        wrapper.vibe_eval("test prompt", return_type=dict)

def test_vibe_eval_api_exception(wrapper, mock_openai_client):
    mock_openai_client.responses.create.side_effect = Exception("API error")

    with pytest.raises(VibeLlmApiException):
        wrapper.vibe_eval("test prompt")
