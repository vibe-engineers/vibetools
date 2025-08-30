"""A wrapper for the Gemini API."""

from typing import Any

from google import genai

from dependencies.llm_wrapper import LlmWrapper
from exceptions import VibeResponseTypeException


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
            response = self.client.models.generate_content(
                model=self.model,
                contents=statement,
                config=genai.types.GenerateContentConfig(system_instruction=self._eval_statement_instruction),
            )

            output_text = response.text.lower().strip()

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

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(system_instruction=self._call_function_instruction),
        )

        return response.text.strip()
