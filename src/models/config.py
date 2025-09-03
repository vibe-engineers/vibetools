"""Additional configurations for VibeCheck."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VibeCheckConfig:
    """
    VibeCheckConfig for additional customizations.

    Notes:
        - Backwards compatibility: `num_tries` is still honored. If both
          `num_tries` and `max_retries` are provided, the wrapper will use
          `max(num_tries, max_retries + 1)` as the total number of attempts.
    """

    # Legacy total-attempts knob (kept for compatibility)
    num_tries: int = 1

    # New retry/backoff knobs
    # max_retries counts additional tries after the first attempt
    max_retries: int = 0
    # Exponential backoff base delay (seconds)
    backoff_base: float = 0.5
    # Maximum backoff delay (seconds)
    backoff_max: float = 8.0
