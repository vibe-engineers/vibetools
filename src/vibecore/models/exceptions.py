"""Custom exceptions for the vibecheck library."""


class VibeLlmClientException(Exception):
    """Raised when the client is not a valid OpenAI or Gemini client."""

    pass


class VibeResponseParseException(Exception):
    """Raised when unable to parse the response from the LLM API to expected type."""

    pass


class VibeLlmApiException(Exception):
    """Raised when LLM API returns an error."""

    pass


class VibeTimeoutException(Exception):
    """Raised when a vibe execution timeouts."""

    pass


class VibeInputTypeException(Exception):
    """Raised when the input type given is invalid."""

    pass
