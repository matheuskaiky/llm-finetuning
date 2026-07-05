"""Unit tests for model providers that do not require the ML stack.

The heavy paths (``load()``) lazily import torch/transformers; here we only check
the pure construction/parameter wiring so the suite stays fast and importable
without torch (see requirements-dev.txt).
"""

from llm_finetuning.models.providers import LocalModelProvider


def test_local_provider_defaults_no_quant():
    p = LocalModelProvider(model_name="some/model")
    assert p.load_in_4bit is False
    assert p.tokenizer_name == "some/model"


def test_local_provider_stores_load_in_4bit():
    p = LocalModelProvider(model_name="some/model", load_in_4bit=True)
    assert p.load_in_4bit is True


def test_local_provider_tokenizer_override():
    p = LocalModelProvider(model_name="m", tokenizer_name="t", load_in_4bit=True)
    assert p.tokenizer_name == "t"
    assert p.load_in_4bit is True
