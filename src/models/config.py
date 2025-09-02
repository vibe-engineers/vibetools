"""Additional configurations for VibeCheck."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VibeCheckConfig:
    """
    VibeCheckConfig for additional customizations.
    """

    num_tries: int = 1
