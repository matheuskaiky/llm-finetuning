"""Tests for the registry and config-driven instantiation."""

from __future__ import annotations

import pytest

from llm_finetuning.core import ComponentSpec, Registry, instantiate


class _Widget:
    def __init__(self, size: int = 1) -> None:
        self.size = size


def _fresh_registry() -> Registry[_Widget]:
    reg: Registry[_Widget] = Registry("widget")
    reg.register("widget")(_Widget)
    return reg


def test_register_and_build() -> None:
    reg = _fresh_registry()
    assert reg.available() == ["widget"]
    widget = reg.build("widget", size=3)
    assert widget.size == 3


def test_duplicate_registration_raises() -> None:
    reg = _fresh_registry()
    with pytest.raises(KeyError):
        reg.register("widget")(_Widget)


def test_unknown_key_raises() -> None:
    reg = _fresh_registry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_instantiate_from_component_spec() -> None:
    reg = _fresh_registry()
    spec = ComponentSpec(type="widget", params={"size": 7})
    widget = instantiate(reg, spec)
    assert widget.size == 7


def test_instantiate_from_plain_dict() -> None:
    reg = _fresh_registry()
    widget = instantiate(reg, {"type": "widget", "params": {"size": 5}})
    assert widget.size == 5
