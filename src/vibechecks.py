import inspect
from typing import Any, Callable
from openai import OpenAI
from google import genai

from dependencies.openai_wrapper import OpenAiWrapper
from dependencies.gemini_wrapper import GeminiWrapper
from exceptions import VibeClientException

class VibeCheck:

    def __init__(self, client: OpenAI | genai.Client, model: str, num_tries: int = 1):
        self.__load_llm(client, model, num_tries)

    def __load_llm(self, client: OpenAI | genai.Client, model: str, num_tries: int):
        if isinstance(client, OpenAI):
            self.llm = OpenAiWrapper(client, model, num_tries)
        elif isinstance(client, genai.Client):
            self.llm = GeminiWrapper(client, model, num_tries)
        else:
            raise VibeClientException("Client must be an instance of openai.OpenAI or genai.Client")

    def __call__(self, arg: str | Callable[..., Any]) -> bool | Callable[..., Any]:
        if isinstance(arg, str):
            return self.llm.vibe_eval_statement(arg)
        elif callable(arg):
            def wrapper(*args, **kwargs):
                func_signature = str(inspect.signature(arg))
                docstring = inspect.getdoc(arg)
                return self.llm.vibe_call_function(func_signature, docstring, *args, **kwargs)
            return wrapper
        else:
            raise VibeInputTypeError("Argument must be a string or a callable")