"""A type-erased public interface for working with LLM clients like OpenAI and Gemini."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai
from openai import OpenAI

from vibecore.llms.gemini_wrapper import GeminiWrapper
from vibecore.llms.llm_wrapper import LlmWrapper
from vibecore.llms.openai_wrapper import OpenAiWrapper
from vibecore.models.exceptions import VibeLlmClientException
from vibecore.models.vibe_config import VibeConfig
from vibecore.utils.logger import ConsoleLogger

if TYPE_CHECKING:

    from google import genai
    from openai import OpenAI

SharedClient = OpenAI | genai.Client


class VibeLlmClient(LlmWrapper):
    """
    Public entrypoint for consumers.

    This class abstracts away the specific LLM client (OpenAI or Gemini),
    and exposes a consistent interface that implements LlmWrapper.

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
        if isinstance(client, OpenAI):
            self.llm = OpenAiWrapper(client, model, config, logger)
            logger.info(f"Loaded OpenAI wrapper with model: {model}")
        elif isinstance(client, genai.Client):
            self.llm = GeminiWrapper(client, model, config, logger)
            logger.info(f"Loaded Gemini wrapper with model: {model}")
        else:
            raise VibeLlmClientException("Client must be an instance of openai.OpenAI or google.genai.Client")

    def vibe_eval_statement(self, statement: str) -> bool:
        """
        Evaluate whether the given statement is true or false using the LLM.

        Args:
            statement (str): A natural-language statement to be evaluated.

        Returns:
            bool: True if the model determines the statement is true, False otherwise.

        """
        return self._run_with_timeout(
            self.llm.vibe_eval_statement,
            self.config.timeout,
            statement,
        )

    def vibe_call_function(self, func_signature, docstring: str, *args, **kwargs):
        """
        Use the LLM to simulate or infer the return value of a function.

        Args:
            func_signature (inspect.Signature): The function's Python signature.
            docstring (str): The function's documentation string.
            *args: Positional arguments to include in the function call.
            **kwargs: Keyword arguments to include in the function call.

        Returns:
            Any: The model's best guess at the return value, potentially coerced into the
                 expected Python return type.

        """
        return self._run_with_timeout(
            self.llm.vibe_call_function,
            self.config.timeout,
            func_signature,
            docstring,
            *args,
            **kwargs,
        )
