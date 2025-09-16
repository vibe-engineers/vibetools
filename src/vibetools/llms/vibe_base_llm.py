"""Abstract base class for LLM wrappers."""

from __future__ import annotations

import inspect
import json
import logging
import queue
import threading
from abc import ABC
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, Optional, Type, get_args, get_origin, get_type_hints

from vibetools.exceptions.exceptions import VibeTimeoutException


class VibeBaseLlm(ABC):
    """Abstract base class for LLM wrappers."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the LLM wrapper.

        Args:
            logger (Optional[logging.Logger]): Logger instance for logging.

        """
        # safe default so abstract classes can log without a child-provided logger
        self.logger = logger or logging.getLogger("vibetools")

    def _run_with_timeout(self, func, timeout, *args, **kwargs):
        """
        Run a function with a timeout.

        Args:
            func: The function to run.
            timeout: The timeout in milliseconds.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.

        Raises:
            VibeTimeoutException: If the function call times out.
            Exception: If the target function raises; the original exception is chained.

        """
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:  # noqa: BLE001
                exception_queue.put(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        # Convert ms -> seconds for join()
        timeout_seconds = timeout / 1000.0
        thread.join(timeout_seconds)

        if thread.is_alive():
            # Still running: timed out
            raise VibeTimeoutException(f"Function call timed out after {timeout} ms")

        # Thread finished: either we have a result or an exception
        if not exception_queue.empty():
            exc = exception_queue.get()
            # Raise a documented Exception type and keep original as __cause__
            raise Exception("Exception occurred in target function") from exc

        # If the function returned None explicitly, result_queue may be empty.
        return result_queue.get_nowait() if not result_queue.empty() else None

    def vibe_eval(self, prompt: str, return_type: Optional[Type] = None) -> Any:
        """
        Evaluate a free-form prompt with LLM and optionally coerce the response.

        - If return_type is None, returns the raw model text (no parsing/formatting).
        - If return_type is a Python type (e.g., str, int, list, dict), the response is
          coerced and validated with the shared helpers.

        Args:
            prompt (str): The prompt to send to the model.
            return_type (Optional[Type]): The expected Python type for coercion. If None,
                the raw text is returned.

        Raises:
            NotImplementedError: This is an abstract method and must be implemented in subclasses.

        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Shared helpers for parsing function response type
    # ---------------------------------------------------------------------

    def _maybe_coerce(self, raw_text: str, expected: Optional[type]) -> Any:
        """
        Attempt to coerce raw LLM text into `expected` when appropriate.

        Args:
            raw_text (str): The raw text response from the LLM.
            expected (Optional[type]): The expected Python type, or None.

        Returns:
            Any: The coerced value if successful; otherwise, the original raw_text.

        """
        if not expected:
            self.logger.debug("_maybe_coerce: no expected type; returning raw_text unchanged")
            return raw_text
        if expected is str:
            self.logger.debug("_maybe_coerce: expected is str; returning raw_text unchanged")
            return raw_text

        # --- Minimal addition: primitive fast-paths (bool/int/float) ---
        if expected in (bool, int, float) and isinstance(raw_text, str):
            s = raw_text.strip()
            if expected is bool:
                low = s.lower()
                if low in {"true", "t", "yes", "y", "1"}:
                    self.logger.debug("_maybe_coerce: primitive bool fast-path → True")
                    return True
                if low in {"false", "f", "no", "n", "0"}:
                    self.logger.debug("_maybe_coerce: primitive bool fast-path → False")
                    return False
                # fall through if not a known boolean literal

            try:
                if expected is int:
                    v = int(s)
                    self.logger.debug("_maybe_coerce: primitive int fast-path")
                    return v
                if expected is float:
                    v = float(s)
                    self.logger.debug("_maybe_coerce: primitive float fast-path")
                    return v
            except Exception:
                # not a plain numeric literal; fall through to JSON
                pass
        # --- end minimal addition ---

        # Only parse JSON when a non-str type is requested.
        try:
            parsed = json.loads(raw_text)
            self.logger.debug(f"_maybe_coerce: JSON parsed successfully (type={type(parsed).__name__})")
        except Exception as e:
            self.logger.debug(f"_maybe_coerce: JSON parse failed: {e}; returning raw_text")
            return raw_text

        # --- Minimal addition: post-JSON nudge for primitives ---
        if expected is bool:
            if isinstance(parsed, bool):
                return parsed
            if isinstance(parsed, str):
                low = parsed.strip().lower()
                if low in {"true", "t", "yes", "y", "1"}:
                    self.logger.debug("_maybe_coerce: post-JSON bool coercion → True")
                    return True
                if low in {"false", "f", "no", "n", "0"}:
                    self.logger.debug("_maybe_coerce: post-JSON bool coercion → False")
                    return False
            if isinstance(parsed, (int, float)) and parsed in (0, 1):
                self.logger.debug("_maybe_coerce: post-JSON numeric→bool coercion")
                return bool(parsed)

        if expected is int and isinstance(parsed, str):
            try:
                self.logger.debug("_maybe_coerce: post-JSON str→int coercion")
                return int(parsed.strip())
            except Exception:
                pass

        if expected is float and isinstance(parsed, str):
            try:
                self.logger.debug("_maybe_coerce: post-JSON str→float coercion")
                return float(parsed.strip())
            except Exception:
                pass
        # --- end minimal addition ---

        # --- Minimal addition: TypedDict handling ---
        if _is_typed_dict(expected) and isinstance(parsed, dict):
            try:
                ann = get_type_hints(expected)
            except Exception:
                ann = getattr(expected, "__annotations__", {}) or {}
            required_keys = set(getattr(expected, "__required_keys__", set()))
            # Keep only annotated keys
            subset = {k: v for k, v in parsed.items() if k in ann}
            # Ensure required keys exist
            if not required_keys.issubset(subset.keys()):
                self.logger.debug("_maybe_coerce: TypedDict missing required keys; returning parsed dict")
                return parsed
            self.logger.debug(
                f"_maybe_coerce: constructing TypedDict {getattr(expected, '__name__', str(expected))} "
                f"from {len(subset)}/{len(parsed)} keys"
            )
            return subset
        # --- end minimal addition ---

        # Dataclasses
        if inspect.isclass(expected) and is_dataclass(expected) and isinstance(parsed, dict):
            try:
                field_names = _dataclass_field_names(expected)
                subset = {k: v for k, v in parsed.items() if k in field_names}
                self.logger.debug(
                    f"_maybe_coerce: constructing dataclass {expected.__name__} "
                    f"from {len(subset)}/{len(parsed)} keys"
                )
                return expected(**subset)
            except Exception as e:
                self.logger.debug(f"_maybe_coerce: dataclass construction failed: {e}; returning parsed dict")
                return parsed  # fall back to parsed JSON; will be validated by _is_match

        # Optional Pydantic (lazy import)
        try:
            from pydantic import BaseModel  # type: ignore

            if inspect.isclass(expected) and issubclass(expected, BaseModel) and isinstance(parsed, dict):
                try:
                    self.logger.debug(f"_maybe_coerce: constructing pydantic model {expected.__name__}")
                    return expected(**parsed)
                except Exception as e:
                    self.logger.debug(f"_maybe_coerce: pydantic construction failed: {e}; returning parsed dict")
                    return parsed
        except Exception:
            # No pydantic installed or import error; silently ignore
            pass

        self.logger.debug(f"_maybe_coerce: returning parsed JSON (type={type(parsed).__name__})")
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
            self.logger.debug("_is_match: no expected type; trivially matches")
            return True
        if expected is str:
            ok = isinstance(value, str)
            self.logger.debug(f"_is_match: expected=str; got={type(value).__name__}; match={ok}")
            return ok

        # Pydantic / dataclass / normal classes
        if isinstance(expected, type):
            try:
                from pydantic import BaseModel  # type: ignore

                if issubclass(expected, BaseModel):
                    ok = isinstance(value, expected)
                    self.logger.debug(
                        f"_is_match: expected=pydantic({expected.__name__}); got={type(value).__name__}; match={ok}"
                    )
                    return ok
            except Exception:
                # pydantic not available; continue
                pass

            # --- Minimal addition: TypedDict validation ---
            if _is_typed_dict(expected):
                if not isinstance(value, dict):
                    self.logger.debug("_is_match: expected=TypedDict; value is not dict; match=False")
                    return False
                try:
                    ann = get_type_hints(expected)
                except Exception:
                    ann = getattr(expected, "__annotations__", {}) or {}
                required_keys = set(getattr(expected, "__required_keys__", set()))
                # Required keys must be present
                if not required_keys.issubset(value.keys()):
                    self.logger.debug("_is_match: TypedDict required keys missing; match=False")
                    return False
                # For keys we know about, validate their types recursively
                for k, t in ann.items():
                    if k in value and not self._is_match(value[k], t):
                        self.logger.debug(f"_is_match: TypedDict field '{k}' type mismatch; match=False")
                        return False
                self.logger.debug("_is_match: TypedDict validation passed; match=True")
                return True
            # --- end minimal addition ---

            if is_dataclass(expected):
                ok = isinstance(value, expected)
                self.logger.debug(
                    f"_is_match: expected=dataclass({expected.__name__}); got={type(value).__name__}; match={ok}"
                )
                return ok

            ok = isinstance(value, expected)
            self.logger.debug(f"_is_match: expected=class({expected.__name__}); got={type(value).__name__}; match={ok}")
            return ok

        # typing constructs (List[T], Dict[K,V], Tuple[...])
        origin = get_origin(expected)
        if origin is None:
            ok = isinstance(value, expected)
            self.logger.debug(
                f"_is_match: expected=typing-unknown; origin=None; got={type(value).__name__}; match={ok}"
            )
            return ok

        if not isinstance(value, origin):
            self.logger.debug(
                f"_is_match: origin={getattr(origin, '__name__', str(origin))}; "
                f"value is not instance of origin; got={type(value).__name__}"
            )
            return False

        args = get_args(expected)
        if origin is list and args:
            (elem_t,) = args
            ok = all(isinstance(x, elem_t) for x in value)
            self.logger.debug(f"_is_match: origin=list; elem_t={getattr(elem_t, '__name__', str(elem_t))}; match={ok}")
            return ok

        if origin is tuple and args:
            if len(args) == len(value):
                ok = all(isinstance(v, t) for v, t in zip(value, args))
                self.logger.debug(f"_is_match: origin=tuple; arity={len(args)}; exact-arity match={ok}")
                return ok
            if len(args) == 2 and args[1] is Ellipsis:
                ok = all(isinstance(v, args[0]) for v in value)
                self.logger.debug(
                    f"_is_match: origin=tuple; variadic_of={getattr(args[0], '__name__', str(args[0]))}; match={ok}"
                )
                return ok
            self.logger.debug(f"_is_match: origin=tuple; arity-mismatch; expected={len(args)}; got={len(value)}")
            return False

        if origin is dict and len(args) == 2:
            k_t, v_t = args
            ok = all(isinstance(k, k_t) and isinstance(v, v_t) for k, v in value.items())
            self.logger.debug(
                f"_is_match: origin=dict; key_t={getattr(k_t, '__name__', str(k_t))}, "
                f"val_t={getattr(v_t, '__name__', str(v_t))}; match={ok}"
            )
            return ok

        self.logger.debug(
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


def _is_typed_dict(tp: Any) -> bool:
    """
    Lightweight runtime check for typing.TypedDict classes.
    Relies on attributes added by the TypedDict metaclass.

    Args:
        tp: The type to check.

    Returns:
        bool: True if tp is a TypedDict class, False otherwise.

    """
    return (
        isinstance(tp, type)
        and hasattr(tp, "__annotations__")
        and hasattr(tp, "__required_keys__")
        and hasattr(tp, "__optional_keys__")
    )
