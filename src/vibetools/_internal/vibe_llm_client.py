"""A type-erased public interface for working with LLM clients like OpenAI and Gemini."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Type

from vibetools._internal.logger import ConsoleLogger
from vibetools._internal.vibe_config import VibeConfig
from vibetools.exceptions import VibeInputTypeException, VibeLlmClientException
from vibetools.llms.gemini_wrapper import GeminiWrapper
from vibetools.llms.openai_wrapper import OpenAiWrapper
from vibetools.llms.vibe_base_llm import VibeBaseLlm

if TYPE_CHECKING:

    from google import genai
    from openai import OpenAI

    SharedClient = OpenAI | genai.Client

# runtime ports for instance checks
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

try:
    from google import genai as _genai
except Exception:
    _genai = None


class VibeLlmClient(VibeBaseLlm):
    """
    Public entrypoint for consumers.

    This class abstracts away the specific LLM client (OpenAI or Gemini),
    and exposes a consistent interface that implements VibeBaseLlm.

    Consumers instantiate this class with a model name, a config, and an LLM client.
    Internally, it dispatches to the appropriate wrapper implementation.
    """

    def __init__(self, client: SharedClient, model: str, config: VibeConfig, logger: ConsoleLogger):
        """
        Initialize the VibeLlmClient with a backend client.

        Args:
            client (SharedClient): Either an instance of `openai.OpenAI` or `google.genai.Client`.
            model (str): The name of the LLM model to use (e.g., "gpt-4", "gemini-pro").
            config (VibeConfig): Configuration options such as retry behavior.
            logger (ConsoleLogger): Logger instance for logging.

        Raises:
            VibeLlmClientException: If the provided client is not a supported type.

        """
        super().__init__(logger)
        # normalize config to vibeconfig if is dict or none and set config
        if isinstance(config, dict) or config is None:
            config = VibeConfig(**(config or {}))
        self.config = config

        # initialize llm based on client type
        if _OpenAI is not None and isinstance(client, _OpenAI):
            self.llm = OpenAiWrapper(client, model, config, logger)
            logger.info(f"Loaded OpenAI wrapper with model: {model}")
        elif _genai is not None and isinstance(client, _genai.Client):
            self.llm = GeminiWrapper(client, model, config, logger)
            logger.info(f"Loaded Gemini wrapper with model: {model}")
        else:
            raise VibeLlmClientException("Client must be an instance of openai.OpenAI or google.genai.Client")

    def vibe_eval(self, prompt: str, return_type: Optional[Type] = None) -> Any:
        """
        Evaluate a free-form prompt with LLM and optionally coerce the response.

        - If return_type is None, returns the raw model text (no parsing/formatting).
        - If return_type is a Python type (e.g., str, int, list, dict), the response is
          coerced and validated with the shared helpers.

        Args:
            prompt (str): The prompt to send to the model.
            return_type (Optional[Type]): The expected Python type for coercion. If None,
                the raw text is returned.

        Returns:
            Any: Raw text if return_type is None; otherwise, the coerced value.

        Raises:
            VibeInputTypeException: If the prompt is not a string.

        """
        if not isinstance(prompt, str):
            raise VibeInputTypeException("Argument must be a string")

        return self._run_with_timeout(self.llm.vibe_eval, self.config.timeout, prompt, return_type)
