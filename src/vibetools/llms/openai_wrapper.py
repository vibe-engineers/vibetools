"""A wrapper for the OpenAI API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from vibetools._internal.logger import ConsoleLogger
from vibetools._internal.vibe_config import VibeConfig
from vibetools.exceptions.exceptions import VibeLlmApiException
from vibetools.llms.vibe_base_llm import VibeBaseLlm


class OpenAiWrapper(VibeBaseLlm):
    """A wrapper for the OpenAI API."""

    def __init__(self, client: OpenAI, model: str, config: VibeConfig, logger: ConsoleLogger):
        """
        Initialize the OpenAI wrapper.

        Args:
            client: The OpenAI client.
            model: The model to use.
            config: VibeConfig containing runtime knobs (e.g., timeout).
            logger (ConsoleLogger): Logger instance for logging.

        """
        super().__init__(config, logger=logger)
        self.client = client
        self.model = model

    def _vibe_eval_llm(self, prompt: str) -> str:
        """
        Evaluate a free-form prompt with OpenAI and return the raw text response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The raw text response from the model.

        Raises:
            VibeLlmApiException: If the LLM API call fails.

        """
        try:
            self.logger.debug(f"Performing _vibe_eval_llm with prompt: {prompt!r}")
            response = self.client.responses.create(
                model=self.model,
                instructions=self.config.system_instruction,
                input=prompt,
            )
            return (getattr(response, "output_text", None) or "").strip()
        except Exception as e:
            raise VibeLlmApiException(f"Unable to evaluate prompt: {e}")
