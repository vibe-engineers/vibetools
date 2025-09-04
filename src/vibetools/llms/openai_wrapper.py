"""A wrapper for the OpenAI API."""

from typing import Any, Optional, Type

from openai import OpenAI

from vibetools._internal.logger import ConsoleLogger
from vibetools.config.vibe_config import VibeConfig
from vibetools.exceptions.exceptions import VibeLlmApiException, VibeResponseParseException
from vibetools.llms.llm_wrapper import LlmWrapper


class OpenAiWrapper(LlmWrapper):
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
            response = self.client.responses.create(
                model=self.model,
                instructions=self._eval_statement_instruction,
                input=prompt,
            )

            raw_text = (getattr(response, "output_text", None) or "").strip()
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
