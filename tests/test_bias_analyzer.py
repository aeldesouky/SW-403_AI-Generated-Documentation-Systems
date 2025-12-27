from src.bias_analyzer import BiasAnalyzer, AccessibilityReport


def test_bias_analyzer_basic_report():
    analyzer = BiasAnalyzer()
    text = (
        "This function adds two numbers and returns the sum. "
        "It handles edge cases like None by raising a ValueError. "
        "Use simple inputs and avoid complex nesting."
    )
    report = analyzer.generate_accessibility_report(text, language="python")
    assert isinstance(report, AccessibilityReport)
    assert 0 <= report.overall_accessibility_score <= 100
    assert report.readability is not None
    assert report.vocabulary is not None
    assert isinstance(report.recommendations, list)
