"""The vibeformat decorator for enforcing response types."""

from utils.logger import console_logger


def vibeformat(*, response_type: type):
    """
    Enforce the response type of functions wrapped by @vibecheck.

    This decorator attaches type expectations to a function. When the function is
    later wrapped by @vibecheck, the underlying LLM wrapper (e.g., GeminiWrapper)
    uses this information to validate and, if possible, coerce the response into
    the specified type. If the response does not match the expected type,
    the LLM wrapper will retry up to its configured `num_tries`.

    Args:
        response_type: The expected Python type of the response. This can be a
            primitive type (e.g., `str`, `int`), a dataclass, or a model type
            such as a Pydantic BaseModel.

    Returns:
        Callable: A decorator that attaches the response type to the target function.

    """

    def _decorator(fn):
        """
        Enforce response type for a function.

        Args:
            fn: The function to decorate.

        Returns:
            function: The same function with `_vibe_response_type` metadata set.

        """
        console_logger.debug(f"Checking response type {response_type} for function '{fn.__name__}'")
        setattr(fn, "_vibe_response_type", response_type)
        return fn

    return _decorator
