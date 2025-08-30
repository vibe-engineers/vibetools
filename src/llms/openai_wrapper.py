"""A wrapper for the OpenAI API."""

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
        for _ in range(0, self.num_tries):
            # TODO: errors can be thrown in this loop, catch it
            console_logger.debug(f"Attempt #{_ + 1}: {statement}")
            response = self.client.responses.create(
                model=self.model,
                instructions=self._eval_statement_instruction,
                input=statement,
            )

            output_text = response.output_text.lower().strip()
            console_logger.debug(f"Response: {output_text}")

            if "true" in output_text:
                return True
            elif "false" in output_text:
                return False

        raise VibeResponseTypeException("Unable to get a valid response from the OpenAI API.")

    def vibe_call_function(self, func_signature: str, docstring: str, *args, **kwargs) -> Any:
        """
        Call a function and returns the result.

        Args:
            func_signature: The function signature.
            docstring: The function's docstring.
            *args: The function's arguments.
            **kwargs: The function's keyword arguments.

        Returns:
            The result of the function call.

        """
        prompt = f"""
        Function Signature: {func_signature}
        Docstring: {docstring}
        Arguments: {args}, {kwargs}
        """
        for _ in range(0, self.num_tries):
            # TODO: errors can be thrown in this loop, catch it
            console_logger.debug(f"Function call prompt: {prompt}")
            response = self.client.responses.create(
                model=self.model,
                instructions=self._call_function_instruction,
                input=prompt,
            )
            console_logger.debug(f"Function call response: {response.output_text.strip()}")

            return response.output_text.strip()
        
        raise VibeResponseTypeException("Unable to get a valid response from the OpenAI API.")
