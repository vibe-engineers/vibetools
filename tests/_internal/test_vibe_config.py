from vibetools._internal.vibe_config import VibeConfig


def test_vibe_config_defaults():
    config = VibeConfig()
    assert config.timeout == 10000
    assert config.system_instruction is None


def test_vibe_config_custom_timeout():
    config = VibeConfig(timeout=5000)
    assert config.timeout == 5000


def test_vibe_config_custom_system_instruction():
    instruction = "You are a helpful assistant."
    config = VibeConfig(system_instruction=instruction)
    assert config.system_instruction == instruction
