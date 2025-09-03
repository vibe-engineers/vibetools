from dataclasses import dataclass

from llms.llm_wrapper import LlmWrapper


class Dummy(LlmWrapper):
    def vibe_eval_statement(self, s: str) -> bool:  # pragma: no cover - not used here
        raise NotImplementedError

    def vibe_call_function(self, *a, **k):  # pragma: no cover - not used here
        raise NotImplementedError


def test_maybe_coerce_json_scalars():
    d = Dummy()
    # Non-str expected → JSON parse is attempted
    assert d._maybe_coerce("8", int) == 8
    assert d._maybe_coerce("3.14", float) == 3.14
    # Expected str short-circuits to raw text
    assert d._maybe_coerce("hello", str) == "hello"


@dataclass
class Point:
    x: int
    y: int


def test_maybe_coerce_dataclass():
    d = Dummy()
    v = d._maybe_coerce('{"x":1,"y":2,"z":9}', Point)
    assert isinstance(v, Point) and v.x == 1 and v.y == 2


def test_maybe_coerce_containers_and_match():
    d = Dummy()
    v = d._maybe_coerce("[1,2,3]", list[int])
    assert v == [1, 2, 3]
    assert d._is_match(v, list[int])

    v2 = d._maybe_coerce('{"a":1,"b":2}', dict[str, int])
    assert v2 == {"a": 1, "b": 2}
    assert d._is_match(v2, dict[str, int])

