import inspect
import pytest

from models.config import VibeCheckConfig
from llms.gemini_wrapper import GeminiWrapper


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_gemini_client(sequence):
    """Create a stub Gemini client.

    sequence: list of Exception or strings (mapped to response.text)
    """
    calls = {"n": 0}

    def generate_content(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        val = sequence[i]
        if isinstance(val, Exception):
            raise val
        return Obj(text=val)

    client = Obj()
    client.models = Obj(generate_content=generate_content)
    client._calls = calls
    return client


def test_statement_api_error_then_success(monkeypatch):
    client = make_gemini_client([RuntimeError("boom"), "true"])
    wrapper = GeminiWrapper(
        client=client,
        model="test-model",
        config=VibeCheckConfig(max_retries=2, backoff_base=0.5, backoff_max=2.0),
    )
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    assert wrapper.vibe_eval_statement("X") is True
    assert client._calls["n"] == 2
    assert sleeps == [0.5]


def test_statement_non_boolean_then_success(monkeypatch):
    client = make_gemini_client(["maybe", "false"])
    wrapper = GeminiWrapper(
        client=client,
        model="test-model",
        config=VibeCheckConfig(max_retries=2, backoff_base=0.25, backoff_max=1.0),
    )
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    assert wrapper.vibe_eval_statement("X") is False
    assert client._calls["n"] == 2
    assert sleeps == [0.25]


def test_statement_exhaust_retries_raises(monkeypatch):
    client = make_gemini_client(["maybe", "maybe", "maybe"])
    wrapper = GeminiWrapper(
        client=client,
        model="test-model",
        config=VibeCheckConfig(max_retries=2, backoff_base=0.5, backoff_max=0.75),
    )
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(Exception):
        wrapper.vibe_eval_statement("X")

    # 3 attempts total → two backoffs
    assert sleeps == [0.5, 0.75]


def test_function_type_mismatch_then_success(monkeypatch):
    # First response cannot be coerced to int; second is JSON number
    client = make_gemini_client(["not an int", "8"])
    wrapper = GeminiWrapper(
        client=client,
        model="test-model",
        config=VibeCheckConfig(max_retries=2, backoff_base=0.1, backoff_max=0.2),
    )

    def f() -> int:  # type: ignore[empty-body]
        ...

    sig = inspect.signature(f)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    out = wrapper.vibe_call_function(sig, "Return an int")
    assert out == 8
    assert sleeps == [0.1]

