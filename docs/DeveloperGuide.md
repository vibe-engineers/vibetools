# Developer Guide

This guide orients you to the codebase, shows how to run examples locally, and explains how to extend VibeChecks with new LLM providers.

## Overview

VibeChecks lets you use LLMs as part of program logic in two ways:
- Statement checks: Pass a string and get a boolean back.
- Function decorators: Decorate a function with a docstring and return type; the LLM “implements” it at runtime.

At runtime, a single facade (`VibeCheck`) chooses the correct provider wrapper (OpenAI or Gemini) and delegates the work.

## Repository Layout

- `src/vibechecks.py`: Facade class that users import. Chooses the wrapper, exposes statement and decorator modes.
- `src/llms/`:
  - `llm_wrapper.py`: Abstract base class + shared helpers for type coercion/validation.
  - `openai_wrapper.py`: OpenAI implementation.
  - `gemini_wrapper.py`: Gemini implementation.
- `src/models/`:
  - `config.py`: `VibeCheckConfig` (e.g., `num_tries`).
  - `exceptions.py`: Custom exception types.
- `src/utils/logger.py`: Colorized console logger; controlled by `VIBE_LOG_LEVEL`.
- `examples/`: Minimal runnable examples for OpenAI and Gemini, using both modes (statement and decorator).
- `tests/`: Placeholder test to confirm PyTest wiring.
- `pyproject.toml`: Tooling (ruff, black, darglint), `pytest` config, hatch scripts.

## Development Setup

Prerequisites:
- Python 3.10+
- API keys as needed: `OPENAI_API_KEY`, `GEMINI_API_KEY`

Recommended steps (PowerShell on Windows):
1) Create and/or activate a virtual environment (repo includes `venv/`):
   - `./venv/Scripts/Activate.ps1`
2) Install the package in editable mode and dev tools:
   - `pip install -e .`
   - `pip install -r requirements-dev.txt` (for `python-dotenv`, linters, pytest)
3) Add a `.env` file at repo root with your keys:
   - `OPENAI_API_KEY=sk-...`
   - `GEMINI_API_KEY=...`
4) Optional verbose logs:
   - `setx VIBE_LOG_LEVEL DEBUG` (new shells) or `$env:VIBE_LOG_LEVEL = "DEBUG"` (current session)

## Running the Examples

From the repo root with the venv active:

OpenAI:
- `python .\examples\openai_statement.py`
- `python .\examples\openai_decorator.py`

Gemini:
- `python .\examples\gemini_statement.py`
- `python .\examples\gemini_decorator.py`

Model tips:
- If a model name isn’t available to your account, swap to a widely available one (e.g., OpenAI: `gpt-4o-mini`; Gemini: `gemini-2.0-flash-lite` or `gemini-1.5-flash`).

## Core Concepts

Facade (`VibeCheck`):
- Initializes with an LLM client and `model`.
- Dual behavior via `__call__`:
  - String → `vibe_eval_statement(...)` → returns `bool`.
  - Callable → returns a wrapper that builds a prompt from the function’s signature, docstring, and arguments; then calls `vibe_call_function(...)`.

Wrappers:
- `OpenAiWrapper` and `GeminiWrapper` implement the same interface using their respective SDKs.
- Both use shared instructions defined in `llm_wrapper.py` for consistency across providers.

Type handling:
- `_maybe_coerce(raw_text, expected)` attempts to parse JSON and construct the expected type (supports dataclasses and, optionally, Pydantic models).
- `_is_match(value, expected)` validates that the coerced value matches the annotated return type, including common typing containers (list, tuple, dict).

Config and logging:
- `VibeCheckConfig(num_tries=1)` controls retry behavior when coercion or responses don’t match expectations.
- Set `VIBE_LOG_LEVEL=DEBUG` to see retry attempts, raw responses, and coercion decisions.

## File-by-File Pointers

- `src/vibechecks.py`:
  - Wrapper selection based on client instance (OpenAI vs Gemini).
  - `__call__` implements string vs decorator logic.
- `src/llms/llm_wrapper.py`:
  - `_eval_statement_instruction` and `_call_function_instruction` prompts.
  - `_maybe_coerce` and `_is_match` for robust type adherence.
- `src/llms/openai_wrapper.py` / `src/llms/gemini_wrapper.py`:
  - Concrete API calls (`responses.create(...)` for OpenAI; `models.generate_content(...)` for Gemini).
- `src/models/config.py`: `VibeCheckConfig` options.
- `src/models/exceptions.py`: Clear failure modes.
- `src/utils/logger.py`: Console logger with color levels.

## Extending: Adding a New Provider

1) Create a new wrapper in `src/llms/<provider>_wrapper.py`:
   - Subclass `VibeBaseLlm`.
   - Implement `vibe_eval_statement(self, statement: str) -> bool` and `vibe_call_function(self, func_signature, docstring, *args, **kwargs) -> Any` using that provider’s SDK.
   - Reuse `self._eval_statement_instruction` and `self._call_function_instruction` for consistent prompts.
2) Update the facade in `src/vibechecks.py` to recognize the new client type and instantiate your wrapper.
3) Add minimal examples in `examples/` showing both statement and decorator usage.
4) Update `requirements.txt` and the `README.md` as needed.

## Example: Structured Return Type (Dataclass)

This pattern shows how the decorator mode can return rich types. Put this in a scratch file and run it like the examples.

```python
from dataclasses import dataclass
from vibechecks import VibeCheck
from openai import OpenAI

@dataclass
class Point:
    x: int
    y: int

client = OpenAI()  # or pass api_key=..., or rely on OPENAI_API_KEY in env
vc = VibeCheck(client, model="gpt-4o-mini")

@vc
def midpoint(a: tuple[int, int], b: tuple[int, int]) -> Point:
    """
    Return the midpoint of points a and b as a JSON object {"x": int, "y": int}.
    """
    pass

print(midpoint((0, 0), (2, 2)))  # -> Point(x=1, y=1)
```

Under the hood, the wrapper asks the LLM to return only the value. `VibeBaseLlm._maybe_coerce` JSON-parses and constructs `Point`, then `_is_match` verifies types.

## Tooling

Common commands (via `pyproject.toml` hatch scripts):
- Lint: `hatch run lint`
- Format: `hatch run format`
- Test: `hatch run test` (or `pytest`)

## Troubleshooting

- Missing `dotenv`: `pip install python-dotenv` or `pip install -r requirements-dev.txt`.
- Authentication errors: ensure `.env` has `OPENAI_API_KEY` / `GEMINI_API_KEY` and you’re running from the repo root so `load_dotenv()` finds it.
- Model errors: switch to an available model (e.g., OpenAI `gpt-4o-mini`, Gemini `gemini-2.0-flash-lite`).
- No output / type mismatch: set `VIBE_LOG_LEVEL=DEBUG` to inspect retries and raw responses.
