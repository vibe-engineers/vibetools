import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from vibetools.llms.gemini_wrapper import GeminiWrapper
from vibetools._internal.vibe_config import VibeConfig
from vibetools._internal.vibe_mode import VibeMode
from vibetools.exceptions.exceptions import VibeResponseParseException, VibeLlmApiException
from google import genai

@pytest.fixture
def mock_gemini_client():
    return MagicMock(spec=genai.Client)

@pytest.fixture
def logger():
    return MagicMock()

@pytest.fixture
def config_chill():
    return VibeConfig(vibe_mode=VibeMode.CHILL)

@pytest.fixture
def config_eager():
    return VibeConfig(vibe_mode=VibeMode.EAGER)

@pytest.fixture
def config_aggressive():
    return VibeConfig(vibe_mode=VibeMode.AGGRESSIVE)

@pytest.fixture
def config_with_instructions():
    return VibeConfig(system_instruction="You are a test assistant.")

@pytest.fixture
def wrapper(mock_gemini_client, config_eager, logger):
    return GeminiWrapper(mock_gemini_client, "test_model", config_eager, logger)

def test_vibe_eval_llm_raw_text(wrapper, mock_gemini_client):
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value="  raw text  ")
    mock_gemini_client.models.generate_content.return_value = mock_response

    result = wrapper._vibe_eval_llm("test prompt")
    assert result == "raw text"
    wrapper.client.models.generate_content.assert_called_once()

def test_vibe_eval_llm_api_exception(wrapper, mock_gemini_client):
    mock_gemini_client.models.generate_content.side_effect = Exception("API error")

    with pytest.raises(VibeLlmApiException):
        wrapper._vibe_eval_llm("test prompt")

def test_system_instruction_is_passed(mock_gemini_client, config_with_instructions, logger):
    wrapper = GeminiWrapper(mock_gemini_client, "test_model", config_with_instructions, logger)
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value="raw text")
    mock_gemini_client.models.generate_content.return_value = mock_response

    wrapper._vibe_eval_llm("test prompt")
    wrapper.client.models.generate_content.assert_called_once_with(
        model="test_model",
        contents="test prompt",
        config={"system_instruction": "You are a test assistant."},
    )

def test_vibe_eval_no_retry_on_success(wrapper):
    with patch.object(wrapper, '_vibe_eval_llm', return_value='{"key": "value"}') as mock_method:
        result = wrapper.vibe_eval("test prompt", return_type=dict)
        assert result == {"key": "value"}
        mock_method.assert_called_once()

def test_vibe_eval_retry_chill(mock_gemini_client, logger, config_chill):
    wrapper = GeminiWrapper(mock_gemini_client, "test_model", config_chill, logger)
    with patch.object(wrapper, '_vibe_eval_llm', side_effect=VibeLlmApiException("API error")) as mock_method:
        with pytest.raises(VibeLlmApiException):
            wrapper.vibe_eval("test prompt")
        assert mock_method.call_count == 1

def test_vibe_eval_retry_eager(mock_gemini_client, logger, config_eager):
    wrapper = GeminiWrapper(mock_gemini_client, "test_model", config_eager, logger)
    with patch.object(wrapper, '_vibe_eval_llm', side_effect=VibeLlmApiException("API error")) as mock_method:
        with pytest.raises(VibeLlmApiException):
            wrapper.vibe_eval("test prompt")
        assert mock_method.call_count == 2

def test_vibe_eval_retry_aggressive(mock_gemini_client, logger, config_aggressive):
    wrapper = GeminiWrapper(mock_gemini_client, "test_model", config_aggressive, logger)
    with patch.object(wrapper, '_vibe_eval_llm', side_effect=VibeLlmApiException("API error")) as mock_method:
        with pytest.raises(VibeLlmApiException):
            wrapper.vibe_eval("test prompt")
        assert mock_method.call_count == 3

def test_vibe_eval_retry_on_parse_error_success(mock_gemini_client, logger, config_eager):
    wrapper = GeminiWrapper(mock_gemini_client, "test_model", config_eager, logger)
    with patch.object(wrapper, '_vibe_eval_llm', side_effect=["not a dict", '{"key": "value"}']) as mock_method:
        result = wrapper.vibe_eval("test prompt", return_type=dict)
        assert result == {"key": "value"}
        assert mock_method.call_count == 2

def test_vibe_eval_fails_after_retries_on_parse_error(mock_gemini_client, logger, config_eager):
    wrapper = GeminiWrapper(mock_gemini_client, "test_model", config_eager, logger)
    with patch.object(wrapper, '_vibe_eval_llm', side_effect=["not a dict", "not a dict either"]) as mock_method:
        with pytest.raises(VibeResponseParseException):
            wrapper.vibe_eval("test prompt", return_type=dict)
        assert mock_method.call_count == 2
