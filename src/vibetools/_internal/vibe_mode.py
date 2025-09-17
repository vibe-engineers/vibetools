"""VibeMode for different retry strategies."""

from enum import Enum


class VibeMode(Enum):
    """
    VibeMode for different retry strategies.

    - CHILL: 1 attempt (no retries)
    - EAGER: 2 attempts (1 retry)
    - AGGRESSIVE: 3 attempts (2 retries)
    """

    CHILL = "chill"
    EAGER = "eager"
    AGGRESSIVE = "aggressive"
