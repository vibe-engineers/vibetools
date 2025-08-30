"""A wrapper for the Gemini API."""

from typing import Any

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
        for _ in range(0, self.num_tries):
            console_logger.debug(f"Attempt #{_ + 1}: {statement}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=statement,
                config=genai.types.GenerateContentConfig(system_instruction=self._eval_statement_instruction),
            )

            output_text = response.text.lower().strip()
            console_logger.debug(f"Response: {output_text}")

            if "true" in output_text:
                return True
            elif "false" in output_text:
                return False

        raise VibeResponseTypeException("Unable to get a valid response from the Gemini API.")

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
        console_logger.debug(f"Function call prompt: {prompt}")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(system_instruction=self._call_function_instruction),
        )
        console_logger.debug(f"Function call response: {response.text.strip()}")

        return response.text.strip()
