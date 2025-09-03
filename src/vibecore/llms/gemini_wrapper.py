"""A wrapper for the Gemini API."""

import inspect
from typing import Any

from google import genai

from vibecore.llms.llm_wrapper import LlmWrapper
from vibecore.models.exceptions import (
    VibeLlmApiException,
    VibeResponseParseException,
)
from vibecore.models.vibe_llm_config import VibeLlmConfig
from vibecore.utils.logger import ConsoleLogger


class GeminiWrapper(LlmWrapper):
    """A wrapper for the Gemini API."""

    def __init__(self, client: genai.Client, model: str, config: VibeLlmConfig, logger: ConsoleLogger):
        """
        Initialize the Gemini wrapper.

        Args:
            client: The Gemini client.
            model: The model to use.
            config: VibeLlmConfig containing runtime knobs (e.g., timeout).
            logger (ConsoleLogger): Logger instance for logging.

        """
        super().__init__(logger=logger)
        self.client = client
        self.model = model
        self.config = config

    def vibe_eval_statement(self, statement: str) -> bool:
        """
        Evaluate a statement and returns a boolean.

        Args:
            statement: The statement to evaluate.

        Returns:
            A boolean indicating whether the statement is true or false.

        Raises:
            VibeResponseParseException: If the API is unable to provide a boolean response.
            VibeLlmApiException: If the LLM API returns an error.

        """
        try:
            self.logger.debug(f"Performing statement evaluation: {statement}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=statement,
                config=genai.types.GenerateContentConfig(system_instruction=self._eval_statement_instruction),
            )

            output_text = (getattr(response, "text", None) or "").lower().strip()
            self.logger.debug(f"Response: {output_text!r}")

            if "true" in output_text:
                return True
            if "false" in output_text:
                return False
            raise VibeResponseParseException("Unable to parse response to expected bool type.")
        except Exception as e:
            raise VibeLlmApiException(f"Unable to evaluate statement: {e}")

    def vibe_call_function(self, func_signature: inspect.signature, docstring: str, *args, **kwargs) -> Any:
        """
        Call a function and return the LLM-evaluated result.

        Builds a structured prompt from the provided signature, docstring, and arguments,
        queries Gemini, and optionally enforces the output's Python type.

        Args:
            func_signature (inspect.signature): The function signature being invoked.
            docstring (str): The function's docstring used to give additional context to the model.
            *args: Positional arguments to include in the call.
            **kwargs: Keyword arguments to include in the call.

        Returns:
            Any: If return type is not found in the function signature, defaults to str.
            Otherwise, returns the value coerced to the return type on success.

        Raises:
            VibeResponseParseException: If the model fails to produce a valid response matching
                the return type (when specified).
            VibeLlmApiException: If the LLM API returns an error.

        """
        if func_signature.return_annotation is inspect.Signature.empty:
            return_type = None
        else:
            return_type = func_signature.return_annotation
        return_type_line = f"\nReturn Type: {return_type}" if return_type else ""
        prompt = f"""
        Function Signature: {func_signature}
        Docstring: {docstring}
        Arguments: {args}, {kwargs}{return_type_line}
        """.strip()

        try:
            self.logger.debug(f"Performing function call: {prompt}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(system_instruction=self._call_function_instruction),
            )

            raw_text = (getattr(response, "text", None) or "").strip()
            self.logger.debug(f"Response: {raw_text!r}")

            # if no return type was specified, default to string
            if return_type is None:
                return raw_text

            # otherwise, enforce the type with shared helpers.
            value = self._maybe_coerce(raw_text, return_type)
            if self._is_match(value, return_type):
                return value

            raise VibeResponseParseException(f"Unable to parse response to expected {return_type!r} type.")
        except Exception as e:
            raise VibeLlmApiException(f"Unable to call function: {e}")
