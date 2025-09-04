import logging
import pytest
from vibetools._internal.logger import ColorFormatter, ConsoleLogger, LEVEL_COLORS, YELLOW, RESET

def test_color_formatter():
    formatter = ColorFormatter("[%(name)s] %(levelname)s: %(message)s")
    record = logging.LogRecord(
        "test_logger", logging.INFO, "/path/to/file", 10, "This is a test message.", (), None
    )
    formatted_message = formatter.format(record)
    expected_name = f"{YELLOW}test_logger{RESET}"
    expected_level = f"{LEVEL_COLORS['INFO']}INFO{RESET}"
    assert expected_name in formatted_message
    assert expected_level in formatted_message
    assert "This is a test message." in formatted_message

def test_console_logger():
    logger = ConsoleLogger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.ERROR  # Default level
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[0].formatter, ColorFormatter)

@pytest.mark.parametrize("level_name, level_num", [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR", logging.ERROR),
    ("CRITICAL", logging.CRITICAL),
])
def test_console_logger_with_level(monkeypatch, level_name, level_num):
    monkeypatch.setenv("VIBE_LOG_LEVEL", level_name)
    # We need to re-import the module to pick up the new env var
    import importlib
    from vibetools._internal import logger
    importlib.reload(logger)
    
    # Now, create the logger, it should use the new level
    reloaded_logger = logger.ConsoleLogger("reloaded_logger")
    assert reloaded_logger.level == level_num
    
    # Clean up by reloading again without the env var
    monkeypatch.delenv("VIBE_LOG_LEVEL")
    importlib.reload(logger)
