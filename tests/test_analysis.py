import src.analysis as analysis


def test_detect_hallucination_uses_cloud_fallback(monkeypatch):
    # Force fallback path by making MultiJudge import fail via detector call
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("BYTEZ_KEY", "")

    # Patch the internal cloud API function to avoid network
    def fake_cloud_detect(source_code, generated_doc, model_name="gpt-4"):
        return {
            "has_hallucination": False,
            "error_type": "No Error",
            "root_cause": "N/A",
        }

    monkeypatch.setattr(analysis, "_detect_with_cloud_api", fake_cloud_detect, raising=True)

    res = analysis.detect_hallucination("def add(a,b): return a+b", "Adds two numbers.")
    assert isinstance(res, dict)
    assert res["has_hallucination"] is False
    assert res["error_type"] == "No Error"
