"""A wrapper for the OpenAI API."""

import inspect
from typing import Any

from openai import OpenAI

from vibecore.llms.llm_wrapper import LlmWrapper
from vibecore.models.exceptions import VibeLlmApiException, VibeResponseParseException
from vibecore.models.vibe_llm_config import VibeLlmConfig
from vibecore.utils.logger import ConsoleLogger


class OpenAiWrapper(LlmWrapper):
    """A wrapper for the OpenAI API."""

    def __init__(self, client: OpenAI, model: str, config: VibeLlmConfig, logger: ConsoleLogger):
        """
        Initialize the OpenAI wrapper.

        Args:
            client: The OpenAI client.
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
            response = self.client.responses.create(
                model=self.model,
                instructions=self._eval_statement_instruction,
                input=statement,
            )

            output_text = (getattr(response, "output_text", None) or "").lower().strip()
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
        queries the OpenAI API, and optionally enforces the output's Python type.

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
            response = self.client.responses.create(
                model=self.model,
                instructions=self._call_function_instruction,
                input=prompt,
            )

            raw_text = (getattr(response, "output_text", None) or "").strip()
            self.logger.debug(f"Function call raw response: {raw_text!r}")

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
