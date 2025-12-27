import types

import pytest

import src.evaluator as evaluator


class FakeScorer:
    def score(self, candidates, references):
        # Return tensors-like; use simple objects with .mean().item()
        class FakeTensor:
            def __init__(self, val):
                self._val = val
            def mean(self):
                return self
            def item(self):
                return self._val
        # P, R, F1
        return FakeTensor(0.8), FakeTensor(0.85), FakeTensor(0.9)


def test_calculate_metrics_monkeypatched_global_scorer(monkeypatch):
    # Avoid heavy model load by patching global_scorer
    monkeypatch.setattr(evaluator, "global_scorer", FakeScorer(), raising=True)
    ref = "Adds two numbers and returns the result."
    cand = "Adds two integers and returns the sum."
    m = evaluator.calculate_metrics(ref, cand)
    assert set(m.keys()) == {"bleu", "rouge_l", "bert_similarity"}
    assert 0.0 <= m["bleu"] <= 1.0
    assert 0.0 <= m["rouge_l"] <= 1.0
    assert m["bert_similarity"] == pytest.approx(0.9)
