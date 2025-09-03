"""Abstract base class for LLM wrappers."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, Optional, get_args, get_origin

from utils.logger import console_logger


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

        Raises:
            VibeResponseParseException: If the model does not return a valid parsed response.

        """
        raise NotImplementedError

    @abstractmethod
    def vibe_call_function(self, func_signature: inspect.signature, docstring: str, *args, **kwargs) -> Any:
        """
        Call a function and returns the result.

        Args:
            func_signature: The function signature.
            docstring: The function's docstring.
            *args: The function's arguments.
            **kwargs: The function's keyword arguments.

        Returns:
            The result of the function call.

        Raises:
            VibeResponseParseException: If the model does not return a valid parsed response.

        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Shared helpers for parsing function response type
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
        if not expected:
            console_logger.debug("_maybe_coerce: no expected type; returning raw_text unchanged")
            return raw_text
        if expected is str:
            console_logger.debug("_maybe_coerce: expected is str; returning raw_text unchanged")
            return raw_text

        # Only parse JSON when a non-str type is requested.
        try:
            parsed = json.loads(raw_text)
            console_logger.debug(f"_maybe_coerce: JSON parsed successfully (type={type(parsed).__name__})")
        except Exception as e:
            console_logger.debug(f"_maybe_coerce: JSON parse failed: {e}; returning raw_text")
            return raw_text

        # Dataclasses
        if inspect.isclass(expected) and is_dataclass(expected) and isinstance(parsed, dict):
            try:
                field_names = _dataclass_field_names(expected)
                subset = {k: v for k, v in parsed.items() if k in field_names}
                console_logger.debug(
                    f"_maybe_coerce: constructing dataclass {expected.__name__} "
                    f"from {len(subset)}/{len(parsed)} keys"
                )
                return expected(**subset)
            except Exception as e:
                console_logger.debug(f"_maybe_coerce: dataclass construction failed: {e}; returning parsed dict")
                return parsed  # fall back to parsed JSON; will be validated by _is_match

        # Optional Pydantic (lazy import)
        try:
            from pydantic import BaseModel  # type: ignore

            if inspect.isclass(expected) and issubclass(expected, BaseModel) and isinstance(parsed, dict):
                try:
                    console_logger.debug(f"_maybe_coerce: constructing pydantic model {expected.__name__}")
                    return expected(**parsed)
                except Exception as e:
                    console_logger.debug(f"_maybe_coerce: pydantic construction failed: {e}; returning parsed dict")
                    return parsed
        except Exception:
            # No pydantic installed or import error; silently ignore
            pass

        console_logger.debug(f"_maybe_coerce: returning parsed JSON (type={type(parsed).__name__})")
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
            console_logger.debug("_is_match: no expected type; trivially matches")
            return True
        if expected is str:
            ok = isinstance(value, str)
            console_logger.debug(f"_is_match: expected=str; got={type(value).__name__}; match={ok}")
            return ok

        # Pydantic / dataclass / normal classes
        if isinstance(expected, type):
            try:
                from pydantic import BaseModel  # type: ignore

                if issubclass(expected, BaseModel):
                    ok = isinstance(value, expected)
                    console_logger.debug(
                        f"_is_match: expected=pydantic({expected.__name__}); got={type(value).__name__}; match={ok}"
                    )
                    return ok
            except Exception:
                # pydantic not available; continue
                pass

            if is_dataclass(expected):
                ok = isinstance(value, expected)
                console_logger.debug(
                    f"_is_match: expected=dataclass({expected.__name__}); got={type(value).__name__}; match={ok}"
                )
                return ok

            ok = isinstance(value, expected)
            console_logger.debug(
                f"_is_match: expected=class({expected.__name__}); got={type(value).__name__}; match={ok}"
            )
            return ok

        # typing constructs (List[T], Dict[K,V], Tuple[...])
        origin = get_origin(expected)
        if origin is None:
            ok = isinstance(value, expected)
            console_logger.debug(
                f"_is_match: expected=typing-unknown; origin=None; got={type(value).__name__}; match={ok}"
            )
            return ok

        if not isinstance(value, origin):
            console_logger.debug(
                f"_is_match: origin={getattr(origin, '__name__', str(origin))}; "
                f"value is not instance of origin; got={type(value).__name__}"
            )
            return False

        args = get_args(expected)
        if origin is list and args:
            (elem_t,) = args
            ok = all(isinstance(x, elem_t) for x in value)
            console_logger.debug(
                f"_is_match: origin=list; elem_t={getattr(elem_t, '__name__', str(elem_t))}; match={ok}"
            )
            return ok

        if origin is tuple and args:
            if len(args) == len(value):
                ok = all(isinstance(v, t) for v, t in zip(value, args))
                console_logger.debug(f"_is_match: origin=tuple; arity={len(args)}; exact-arity match={ok}")
                return ok
            if len(args) == 2 and args[1] is Ellipsis:
                ok = all(isinstance(v, args[0]) for v in value)
                console_logger.debug(
                    f"_is_match: origin=tuple; variadic_of={getattr(args[0], '__name__', str(args[0]))}; match={ok}"
                )
                return ok
            console_logger.debug(f"_is_match: origin=tuple; arity-mismatch; expected={len(args)}; got={len(value)}")
            return False

        if origin is dict and len(args) == 2:
            k_t, v_t = args
            ok = all(isinstance(k, k_t) and isinstance(v, v_t) for k, v in value.items())
            console_logger.debug(
                f"_is_match: origin=dict; key_t={getattr(k_t, '__name__', str(k_t))}, "
                f"val_t={getattr(v_t, '__name__', str(v_t))}; match={ok}"
            )
            return ok

        console_logger.debug(
            f"_is_match: origin={getattr(origin, '__name__', str(origin))}; " f"no specific handler; defaulting to True"
        )
        return True


@lru_cache(maxsize=256)
def _dataclass_field_names(dc: type) -> set[str]:
    """
    Cache dataclass field names to avoid repeated reflection.

    Args:
        dc: The dataclass type to inspect.

    Returns:
        set[str]: A set of field names for the dataclass.

    """
    return {f.name for f in fields(dc)}
