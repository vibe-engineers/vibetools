from vibetools._internal.vibe_config import VibeConfig
from vibetools._internal.vibe_mode import VibeMode
import pytest

def test_vibe_config_defaults():
    config = VibeConfig()
    assert config.timeout == 10000
    assert config.system_instruction is None
    assert config.vibe_mode == VibeMode.CHILL


def test_vibe_config_custom_timeout():
    config = VibeConfig(timeout=5000)
    assert config.timeout == 5000


def test_vibe_config_custom_system_instruction():
    instruction = "You are a helpful assistant."
    config = VibeConfig(system_instruction=instruction)
    assert config.system_instruction == instruction


def test_vibe_config_vibe_mode_string_upper():
    config = VibeConfig(vibe_mode="AGGRESSIVE")
    assert config.vibe_mode == VibeMode.AGGRESSIVE


def test_vibe_config_vibe_mode_string_lower():
    config = VibeConfig(vibe_mode="chill")
    assert config.vibe_mode == VibeMode.CHILL


def test_vibe_config_vibe_mode_enum():
    config = VibeConfig(vibe_mode=VibeMode.EAGER)
    assert config.vibe_mode == VibeMode.EAGER


def test_vibe_config_vibe_mode_invalid_string():
    with pytest.raises(ValueError, match="Invalid vibe_mode"):
        VibeConfig(vibe_mode="INVALID_MODE")
