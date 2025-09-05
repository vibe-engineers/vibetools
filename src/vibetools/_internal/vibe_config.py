"""Additional configurations for all libraries."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class VibeConfig:
    """
    VibeConfig for additional customizations.
    """

    # system instruction to use for eval calls
    system_instruction: str = None

    # default timeout for LLM calls, in milliseconds
    timeout: int = 10000
