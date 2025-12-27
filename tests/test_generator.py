import types

import pytest

import src.generator as generator


class FakeInferenceClient:
    def __init__(self, model=None, token=None):
        self.model = model
        self.token = token

    def text_generation(self, prompt, temperature=0.0, max_new_tokens=50):
        assert "Docstring:" in prompt
        return "Adds two numbers and returns the sum."


def test_generate_documentation_huggingface_monkeypatch(monkeypatch):
    # Force provider to Hugging Face and stub client
    monkeypatch.setattr(generator, "InferenceClient", FakeInferenceClient, raising=True)
    # Ensure safety functions behave
    doc = generator.generate_documentation(
        code_snippet="def add(a,b): return a+b",
        language="Python",
        model_name="google/flan-t5-small",
        temperature=0.2,
        provider="Hugging Face",
    )
    assert isinstance(doc, str)
    assert "Sorry, I can't assist" not in doc
    assert "sum" in doc.lower()
