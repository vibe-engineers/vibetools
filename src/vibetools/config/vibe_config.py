"""Additional configurations for all libraries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VibeConfig:
    """
    VibeConfig for additional customizations.
    """

    timeout: int = 10000
