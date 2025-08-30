"""The main VibeCheck class and its functionalities."""

import inspect
from typing import Any, Callable

from google import genai
from openai import OpenAI

from llms.gemini_wrapper import GeminiWrapper
from llms.openai_wrapper import OpenAiWrapper
from models.exceptions import VibeClientException, VibeInputTypeException


class VibeCheck:
    """
    A class that uses LLMs to perform "vibe checks" on statements or function calls.

    This class can be used as a decorator or as a function call to evaluate
    the logical validity of a statement or the outcome of a function call.
    """

    def __init__(self, client: OpenAI | genai.Client, model: str, num_tries: int = 1):
        """
        Initialize the VibeCheck object.

        Args:
            client: An instance of `openai.OpenAI` or `genai.Client`.
            model: The name of the model to use for the LLM.
            num_tries: The number of times to try the LLM call if it fails.

        """
        self.__load_llm(client, model, num_tries)

    def __load_llm(self, client: OpenAI | genai.Client, model: str, num_tries: int):
        """
        Load the appropriate LLM wrapper based on the client type.

        Args:
            client: An instance of `openai.OpenAI` or `genai.Client`.
            model: The name of the model to use for the LLM.
            num_tries: The number of times to try the LLM call if it fails.

        Raises:
            VibeClientException: If the client is not a valid OpenAI or Gemini client.

        """
        if isinstance(client, OpenAI):
            self.llm = OpenAiWrapper(client, model, num_tries)
        elif isinstance(client, genai.Client):
            self.llm = GeminiWrapper(client, model, num_tries)
        else:
            raise VibeClientException("Client must be an instance of openai.OpenAI or genai.Client")

    def __call__(self, arg: str | Callable[..., Any]) -> bool | Callable[..., Any]:
        """
        Perform a vibe check on a statement or a function call.

        If the argument is a string, it evaluates the statement's truthiness.
        If the argument is a callable, it wraps the function to evaluate its
        intended outcome based on its docstring and arguments.

        Args:
            arg: A string statement or a callable function.

        Returns:
            If `arg` is a string, returns a boolean.
            If `arg` is a callable, returns a wrapped function.

        Raises:
            VibeInputTypeException: If the argument is not a string or a callable.

        """
        if isinstance(arg, str):
            return self.llm.vibe_eval_statement(arg)
        elif callable(arg):

            def wrapper(*args, **kwargs):
                func_signature = str(inspect.signature(arg))
                docstring = inspect.getdoc(arg)
                return self.llm.vibe_call_function(func_signature, docstring, *args, **kwargs)

            return wrapper
        else:
            raise VibeInputTypeException("Argument must be a string or a callable")
