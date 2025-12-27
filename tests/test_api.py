from fastapi.testclient import TestClient
import sys
import types

# Prepare stubs for heavy dependencies used by API when optional features are enabled
fake_evaluator = types.ModuleType("src.evaluator")

def fake_calculate_metrics(reference, candidate):
    return {"bleu": 0.5, "rouge_l": 0.6, "bert_similarity": 0.7}

fake_evaluator.calculate_metrics = fake_calculate_metrics

fake_analysis = types.ModuleType("src.analysis")

def fake_detect_hallucination(code, doc):
    return {"has_hallucination": False, "error_type": "No Error", "root_cause": "N/A"}

fake_analysis.detect_hallucination = fake_detect_hallucination

fake_bias = types.ModuleType("src.bias_analyzer")

class FakeReadability:
    flesch_reading_ease = 60.0
    flesch_kincaid_grade = 8.0
    gunning_fog = 10.0

class FakeReport:
    overall_accessibility_score = 75.0
    grade_level_interpretation = "Middle school level"
    readability = FakeReadability()
    recommendations = ["Use simpler words", "Shorten sentences"]

class FakeAnalyzer:
    def generate_accessibility_report(self, text, lang):
        return FakeReport()

fake_bias.BiasAnalyzer = FakeAnalyzer

sys.modules["src.evaluator"] = fake_evaluator
sys.modules["src.analysis"] = fake_analysis
sys.modules["src.bias_analyzer"] = fake_bias

# Import after stubbing
from src.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_minimal():
    # Stub generator to avoid network
    import src.api as api_mod

    def fake_generate_documentation(**kwargs):
        return "Adds two numbers and returns the sum."

    api_mod.generate_documentation = fake_generate_documentation

    payload = {
        "code_snippet": "def add(a,b): return a+b",
        "language": "Python",
        "provider": "Hugging Face",
        "model_name": "google/flan-t5-small",
        "temperature": 0.2,
        "ground_truth": "Add two numbers and return sum.",
        "run_hallucination_check": True,
        "run_accessibility_analysis": True,
    }
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "docstring" in data and isinstance(data["docstring"], str)
    assert data["language"] == "Python"
    assert "metrics" in data and isinstance(data["metrics"], dict)
    assert "hallucination" in data and isinstance(data["hallucination"], dict)
    assert "accessibility" in data and isinstance(data["accessibility"], dict)
