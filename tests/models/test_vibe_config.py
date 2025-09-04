from vibetools.models.vibe_config import VibeConfig

def test_vibe_config_defaults():
    config = VibeConfig()
    assert config.timeout == 10000

def test_vibe_config_custom_timeout():
    config = VibeConfig(timeout=5000)
    assert config.timeout == 5000
