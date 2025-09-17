"""A wrapper for the Gemini API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai

from vibetools._internal.logger import ConsoleLogger
from vibetools._internal.vibe_config import VibeConfig
from vibetools.exceptions.exceptions import VibeLlmApiException
from vibetools.llms.vibe_base_llm import VibeBaseLlm


class GeminiWrapper(VibeBaseLlm):
    """A wrapper for the Gemini API."""

    def __init__(self, client: genai.Client, model: str, config: VibeConfig, logger: ConsoleLogger):
        """
        Initialize the Gemini wrapper.

        Args:
            client: The Gemini client.
            model: The model to use.
            config: VibeConfig containing runtime knobs (e.g., timeout).
            logger (ConsoleLogger): Logger instance for logging.

        """
        super().__init__(config, logger=logger)
        self.client = client
        self.model = model

    def _vibe_eval_llm(self, prompt: str) -> str:
        """
        Evaluate a free-form prompt with Gemini and return the raw text response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The raw text response from the model.

        Raises:
            VibeLlmApiException: If the LLM API call fails.

        """
        try:
            self.logger.debug(f"Performing _vibe_eval_llm with prompt: {prompt!r}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"system_instruction": self.config.system_instruction},
            )

            return (getattr(response, "text", None) or "").strip()
        except Exception as e:
            raise VibeLlmApiException(f"Unable to evaluate prompt: {e}")
