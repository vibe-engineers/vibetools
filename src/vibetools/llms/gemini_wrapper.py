"""A wrapper for the Gemini API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    from google import genai

from vibetools._internal.logger import ConsoleLogger
from vibetools._internal.vibe_config import VibeConfig
from vibetools.exceptions.exceptions import (
    VibeLlmApiException,
    VibeResponseParseException,
)
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
        super().__init__(logger=logger)
        self.client = client
        self.model = model
        self.config = config

    def vibe_eval(self, prompt: str, return_type: Optional[Type] = None) -> Any:
        """
        Evaluate a free-form prompt with Gemini and optionally coerce the response.

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
            VibeResponseParseException: If coercion is requested but fails.
            VibeLlmApiException: If the LLM API call fails.

        """
        try:
            self.logger.debug(f"Performing vibe_eval with prompt: {prompt!r}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"system_instruction": self.config.system_instruction},
            )

            raw_text = (getattr(response, "text", None) or "").strip()
            self.logger.debug(f"Raw response: {raw_text!r}")

            if return_type is None:
                return raw_text

            value = self._maybe_coerce(raw_text, return_type)
            if self._is_match(value, return_type):
                return value

            raise VibeResponseParseException(f"Unable to parse response to expected {return_type!r} type.")
        except VibeResponseParseException:
            raise
        except Exception as e:
            raise VibeLlmApiException(f"Unable to evaluate prompt: {e}")
