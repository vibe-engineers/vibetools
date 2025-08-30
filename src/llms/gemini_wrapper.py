"""A wrapper for the Gemini API."""

from typing import Any, Optional

from google import genai

from llms.llm_wrapper import LlmWrapper
from models.exceptions import VibeResponseTypeException
from utils.logger import console_logger


class GeminiWrapper(LlmWrapper):
    """A wrapper for the Gemini API."""

    def __init__(self, client: genai.Client, model: str, num_tries: int):
        """
        Initialize the Gemini wrapper.

        Args:
            client: The Gemini client.
            model: The model to use.
            num_tries: The number of times to try the request.

        """
        self.client = client
        self.model = model
        self.num_tries = num_tries

    def vibe_eval_statement(self, statement: str) -> bool:
        """
        Evaluate a statement and returns a boolean.

        Args:
            statement: The statement to evaluate.

        Returns:
            A boolean indicating whether the statement is true or false.

        Raises:
            VibeResponseTypeException: If the API is unable to provide a valid response.

        """
        for attempt in range(1, self.num_tries + 1):
            # TODO: errors can be thrown in this loop, catch it
            console_logger.debug(f"[Attempt {attempt}/{self.num_tries}] {statement}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=statement,
                config=genai.types.GenerateContentConfig(system_instruction=self._eval_statement_instruction),
            )

            output_text = response.text.lower().strip()
            console_logger.debug(f"Response: {output_text}")

            if "true" in output_text:
                return True
            if "false" in output_text:
                return False

        raise VibeResponseTypeException("Unable to get a valid response from the Gemini API.")

    def vibe_call_function(
        self, func_signature: str, docstring: str, *args, response_type: Optional[type] = None, **kwargs
    ) -> Any:
        """
        Call a function and returns the result.

        Args:
            func_signature: The function signature.
            docstring: The function's docstring.
            *args: The function's arguments.
            response_type: (Optional) Expected Python type for the response. If provided,
                the wrapper will validate (and attempt to coerce) LLM output to this type.
            **kwargs: The function's keyword arguments.

        Returns:
            The result of the function call.

        Raises:
            VibeResponseTypeException: If the API is unable to provide a valid response
            that matches the expected type within the configured number of tries.

        """
        return_type_line = (
            f"\nReturn Type: {getattr(response_type, '__name__', str(response_type))}" if response_type else ""
        )
        prompt = f"""
        Function Signature: {func_signature}
        Docstring: {docstring}
        Arguments: {args}, {kwargs}{return_type_line}
        """.strip()

        for attempt in range(1, self.num_tries + 1):
            # TODO: errors can be thrown in this loop, catch and continue if desired
            console_logger.debug(f"[Attempt {attempt}/{self.num_tries}] Function call prompt: {prompt}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(system_instruction=self._call_function_instruction),
            )
            raw_text = response.text.strip()
            console_logger.debug(f"Function call raw response: {raw_text}")

            # if no response_type was specified (no @vibeformat), keep behavior unchanged.
            if response_type is None:
                return raw_text

            # otherwise, enforce the type with shared helpers.
            value = self._maybe_coerce(raw_text, response_type)
            if self._is_match(value, response_type):
                return value

            console_logger.debug("Response did not match expected type; retrying...")

        raise VibeResponseTypeException(
            f"Unable to get a valid response matching {response_type!r} from the Gemini API."
        )
