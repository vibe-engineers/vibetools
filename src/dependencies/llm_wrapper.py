"""Abstract base class for LLM wrappers."""

from abc import ABC, abstractmethod
from typing import Any


class LlmWrapper(ABC):
    """Abstract base class for LLM wrappers."""

    _eval_statement_instruction = "Evaluate the statement below and respond with either 'true' or 'false'."
    _call_function_instruction = (
        "You will be given: a function signature (name, parameters, and return type); "
        "a docstring describing what the function is intended to do; "
        "the concrete arguments passed to the function; and the declared return value type. "
        "Your task is to: (1) interpret the docstring to understand the intended behavior of the function, "
        "(2) use the provided arguments to simulate what the function would logically produce, "
        "(3) ensure your response strictly matches the declared return type, both in structure and data type, and "
        "(4) return only the value that fulfills the function’s contract, "
        "with no explanations, commentary, or extra text."
    )

    @abstractmethod
    def vibe_eval_statement(self, statement: str) -> bool:
        """
        Evaluate a statement and returns a boolean.

        Args:
            statement: The statement to evaluate.

        Returns:
            A boolean indicating whether the statement is true or false.

        """
        pass

    @abstractmethod
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
        pass
