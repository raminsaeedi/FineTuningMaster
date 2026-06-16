"""Registry behavior and that all four methods are registered."""

import pytest

from src.core.registry import Registry


def test_register_and_get():
    reg = Registry("demo")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert "a" in reg
    assert reg.keys() == ["a"]


def test_duplicate_raises():
    reg = Registry("demo")

    @reg.register("a")
    class A:
        pass

    with pytest.raises(ValueError):
        @reg.register("a")
        class B:
            pass


def test_unknown_key_raises():
    reg = Registry("demo")
    with pytest.raises(KeyError):
        reg.get("missing")


def test_all_methods_registered():
    import src.methods  # noqa: F401
    from src.core.registry import METHODS

    assert set(METHODS.keys()) == {"prompt_only", "ft", "rag", "ft_rag"}
