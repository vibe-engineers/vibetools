import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from vibetools.llms.vibe_base_llm import VibeBaseLlm, _dataclass_field_names
from vibetools.exceptions.exceptions import VibeTimeoutException

# A concrete implementation of the abstract VibeBaseLlm for testing
class ConcreteVibeBaseLlm(VibeBaseLlm):
    def vibe_eval(self, prompt: str, return_type=None):
        pass  # Not needed for these tests

@pytest.fixture
def wrapper():
    return ConcreteVibeBaseLlm(logger=MagicMock())

def test_run_with_timeout_success(wrapper):
    def target_func():
        return "success"
    result = wrapper._run_with_timeout(target_func, 100)
    assert result == "success"

def test_run_with_timeout_timeout(wrapper):
    def target_func():
        import time
        time.sleep(0.2)
        return "should not return"
    with pytest.raises(VibeTimeoutException):
        wrapper._run_with_timeout(target_func, 100)

def test_run_with_timeout_exception(wrapper):
    def target_func():
        raise ValueError("test error")
    with pytest.raises(Exception, match="Exception occurred in target function"):
        wrapper._run_with_timeout(target_func, 100)

@pytest.mark.parametrize("raw_text, expected_type, expected_value", [
    ("true", bool, True),
    ("1", int, 1),
    ("1.1", float, 1.1),
    ('{"a": 1}', dict, {"a": 1}),
    ('[1, 2]', list, [1, 2]),
    ('"test"', str, '"test"'),
])
def test_maybe_coerce(wrapper, raw_text, expected_type, expected_value):
    coerced_value = wrapper._maybe_coerce(raw_text, expected_type)
    assert coerced_value == expected_value

@dataclass
class MyDataClass:
    a: int
    b: str

def test_maybe_coerce_dataclass(wrapper):
    raw_text = '{"a": 1, "b": "test", "c": 3}'
    coerced_value = wrapper._maybe_coerce(raw_text, MyDataClass)
    assert isinstance(coerced_value, MyDataClass)
    assert coerced_value.a == 1
    assert coerced_value.b == "test"

# Basic Pydantic model for testing, if pydantic is installed
try:
    from pydantic import BaseModel
    class MyPydanticModel(BaseModel):
        a: int
        b: str
    PYDANTIC_INSTALLED = True
except ImportError:
    PYDANTIC_INSTALLED = False

@pytest.mark.skipif(not PYDANTIC_INSTALLED, reason="pydantic not installed")
def test_maybe_coerce_pydantic(wrapper):
    raw_text = '{"a": 1, "b": "test"}'
    coerced_value = wrapper._maybe_coerce(raw_text, MyPydanticModel)
    assert isinstance(coerced_value, MyPydanticModel)
    assert coerced_value.a == 1
    assert coerced_value.b == "test"

@pytest.mark.parametrize("value, expected_type, expected_match", [
    ("test", str, True),
    (1, int, True),
    (1.1, float, True),
    (True, bool, True),
    ([1, 2], list[int], True),
    ({"a": 1}, dict[str, int], True),
    ((1, "a"), tuple[int, str], True),
    (MyDataClass(a=1, b="t"), MyDataClass, True),
    ([1, "a"], list[int], False),
])
def test_is_match(wrapper, value, expected_type, expected_match):
    match = wrapper._is_match(value, expected_type)
    assert match == expected_match

def test_dataclass_field_names():
    names = _dataclass_field_names(MyDataClass)
    assert names == {"a", "b"}
