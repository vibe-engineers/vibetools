"""Abstract base class for LLM wrappers."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, Optional, get_args, get_origin


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
        raise NotImplementedError

    @abstractmethod
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

        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Shared helpers (lightweight; only used if response_type is provided)
    # ---------------------------------------------------------------------

    def _maybe_coerce(self, raw_text: str, expected: Optional[type]) -> Any:
        """
        Attempt to coerce raw LLM text into `expected` when appropriate.

        Fast paths:
            - If `expected` is falsy or `str`, return `raw_text` unchanged.

        Otherwise:
            - Try JSON parse; on failure, return `raw_text` (validation will fail upstream).
            - If `expected` is a dataclass and parsed is dict, construct from overlapping keys.
            - If `expected` is a Pydantic BaseModel (if available) and parsed is dict, construct via model(**parsed).
            - For typing containers (list[T], dict[K, V]), return parsed JSON and let `_is_match` validate.

        Args:
            raw_text: The raw text returned by the LLM.
            expected: The expected Python type, or None.

        Returns:
            The coerced value when possible, otherwise the original raw_text or parsed JSON.

        """
        if not expected or expected is str:
            return raw_text

        # Only parse JSON when a non-str type is requested.
        try:
            parsed = json.loads(raw_text)
        except Exception:
            return raw_text

        # Dataclasses
        if inspect.isclass(expected) and is_dataclass(expected) and isinstance(parsed, dict):
            try:
                return expected(**{k: v for k, v in parsed.items() if k in _dataclass_field_names(expected)})
            except Exception:
                return parsed  # fall back to parsed JSON; will be validated by _is_match

        # Optional Pydantic (lazy import)
        try:
            from pydantic import BaseModel  # type: ignore

            if inspect.isclass(expected) and issubclass(expected, BaseModel) and isinstance(parsed, dict):
                try:
                    return expected(**parsed)
                except Exception:
                    return parsed
        except Exception:
            pass

        return parsed

    def _is_match(self, value: Any, expected: Optional[type]) -> bool:
        """
        Check whether `value` conforms to `expected`.

        Fast paths:
            - If `expected` is falsy → True.
            - If `expected` is `str` → isinstance(value, str).

        Handles:
            - Concrete classes (incl. dataclasses, optional Pydantic).
            - Common typing forms: list[T], tuple[T, ...] and tuple[T, ...] ellipsis, dict[K, V].

        Args:
            value: The value to check.
            expected: The expected Python type, or None.

        Returns:
            True if the value matches expectations, False otherwise.

        """
        if not expected:
            return True
        if expected is str:
            return isinstance(value, str)

        # Pydantic / dataclass / normal classes
        if isinstance(expected, type):
            try:
                from pydantic import BaseModel  # type: ignore

                if issubclass(expected, BaseModel):
                    return isinstance(value, expected)
            except Exception:
                pass
            if is_dataclass(expected):
                return isinstance(value, expected)
            return isinstance(value, expected)

        # typing constructs (List[T], Dict[K,V], Tuple[...])
        origin = get_origin(expected)
        if origin is None:
            return isinstance(value, expected)

        if not isinstance(value, origin):
            return False

        args = get_args(expected)
        if origin is list and args:
            (elem_t,) = args
            return all(isinstance(x, elem_t) for x in value)
        if origin is tuple and args:
            if len(args) == len(value):
                return all(isinstance(v, t) for v, t in zip(value, args))
            if len(args) == 2 and args[1] is Ellipsis:
                return all(isinstance(v, args[0]) for v in value)
            return False
        if origin is dict and len(args) == 2:
            k_t, v_t = args
            return all(isinstance(k, k_t) and isinstance(v, v_t) for k, v in value.items())

        return True


@lru_cache(maxsize=256)
def _dataclass_field_names(dc: type) -> set[str]:
    """Cache dataclass field names to avoid repeated reflection."""
    return {f.name for f in fields(dc)}
