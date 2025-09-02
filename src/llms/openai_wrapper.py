"""A wrapper for the OpenAI API."""

import inspect
from typing import Any

from openai import OpenAI

from llms.llm_wrapper import LlmWrapper
from models.exceptions import VibeResponseTypeException
from utils.logger import console_logger


class OpenAiWrapper(LlmWrapper):
    """A wrapper for the OpenAI API."""

    def __init__(self, client: OpenAI, model: str, num_tries: int):
        """
        Initialize the OpenAI wrapper.

        Args:
            client: The OpenAI client.
            model: The model to use.
            num_tries: The number of times to try the request.

        """
        self.client = client
        self.model = model
        self.num_tries = num_tries  # keep parity with your current GeminiWrapper

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
            # catch any error thrown in this loop, log at debug, and retry
            try:
                console_logger.debug(f"[Attempt {attempt}/{self.num_tries}] {statement}")
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self._eval_statement_instruction,
                    input=statement,
                )
            except Exception as e:
                console_logger.debug(f"Error on attempt {attempt}: {e}")
                continue

            output_text = (getattr(response, "output_text", None) or "").lower().strip()
            console_logger.debug(f"Response: {output_text!r}")

            if "true" in output_text:
                return True
            if "false" in output_text:
                return False

        raise VibeResponseTypeException("Unable to get a valid response from the OpenAI API.")

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
            VibeResponseTypeException: If the model fails to produce a valid response matching
                the return type (when specified) within the configured number of tries.

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

        for attempt in range(1, self.num_tries + 1):
            # handle API exceptions and continue if desired
            try:
                console_logger.debug(f"[Attempt {attempt}/{self.num_tries}] Function call prompt: {prompt}")
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self._call_function_instruction,
                    input=prompt,
                )
            except Exception as e:
                console_logger.debug(f"Error on attempt {attempt}: {e}")
                continue

            raw_text = (getattr(response, "output_text", None) or "").strip()
            console_logger.debug(f"Function call raw response: {raw_text!r}")

            # if no return type was specified, default to string
            if return_type is None:
                return raw_text

            # otherwise, enforce the type with shared helpers.
            value = self._maybe_coerce(raw_text, return_type)
            if self._is_match(value, return_type):
                return value

            console_logger.debug("Response did not match expected type; retrying...")

        raise VibeResponseTypeException(f"Unable to get a valid response matching {return_type!r} from the OpenAI API.")
