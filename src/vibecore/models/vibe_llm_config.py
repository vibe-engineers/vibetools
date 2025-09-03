"""Additional configurations for VibeCheck."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VibeLlmConfig:
    """
    VibeLlmConfig for additional customizations.
    """

    timeout: int = 10000
