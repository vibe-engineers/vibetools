"""Additional configurations for all libraries."""

from dataclasses import dataclass

from vibetools._internal.vibe_mode import VibeMode


@dataclass(frozen=True, kw_only=True)
class VibeConfig:
    """
    VibeConfig for additional customizations.
    """

    # system instruction to use for eval calls
    system_instruction: str = None

    # default timeout for LLM calls, in milliseconds
    timeout: int = 10000

    # retry mode for vibe_eval calls
    vibe_mode: VibeMode | str = VibeMode.CHILL

    def __post_init__(self):
        if isinstance(self.vibe_mode, str):
            try:
                mode = VibeMode[self.vibe_mode.upper()]
                object.__setattr__(self, "vibe_mode", mode)
            except KeyError:
                valid_modes = [mode.name for mode in VibeMode]
                raise ValueError(f"Invalid vibe_mode '{self.vibe_mode}'. Valid modes are: {valid_modes}")
