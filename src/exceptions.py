class VibeClientException(Exception):
    """Raised when the client is not a valid OpenAI or Gemini client."""
    pass

class VibeResponseTypeException(Exception):
    """Raised when the response type given is invalid."""
    pass

class VibeInputTypeError(Exception):
    """Raised when the input type given is invalid."""
    pass
