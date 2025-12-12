"""
Hallucination Detection Module
==============================

This module provides hallucination detection for AI-generated documentation.
Uses a multi-judge ensemble with 3 local Ollama models for reliable detection.

Models:
- qwen2.5-coder:7b (Code Searcher) - Finds code discrepancies
- deepseek-r1:7b (Reasoner) - Logical step-by-step analysis
- gemma3:4b (Thinker) - Holistic pattern recognition

Author: AI Documentation Systems Team
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()


def detect_hallucination(source_code: str, generated_doc: str, 
                         model_name: str = None) -> Dict[str, Any]:
    """
    Detect hallucinations in AI-generated documentation using multi-judge ensemble.
    
    Uses 3 local Ollama models that vote on whether the documentation contains
    hallucinations, fabrications, or omissions.
    
    Args:
        source_code: The original source code being documented.
        generated_doc: The AI-generated documentation to verify.
        model_name: Ignored (kept for backward compatibility).
    
    Returns:
        Dictionary containing:
        - has_hallucination: bool - True if hallucination detected
        - error_type: str - Type of error (Fabricated Variable, Wrong Logic, etc.)
        - root_cause: str - Explanation of the issue
        - confidence: float - Confidence level (0.0 to 1.0)
        - agreement: float - Agreement ratio between models
        - votes: list - Individual model votes
    """
    try:
        from multi_judge import MultiJudgeHallucinationDetector
        
        detector = MultiJudgeHallucinationDetector(auto_pull_models=True)
        result = detector.detect(source_code, generated_doc, parallel=True)
        
        return {
            "has_hallucination": result.final_verdict,
            "error_type": result.error_type,
            "root_cause": result.root_cause,
            "confidence": result.confidence,
            "agreement": result.agreement_ratio,
            "votes": [
                {
                    "model": v.model,
                    "role": v.role.value,
                    "verdict": v.has_hallucination,
                    "error_type": v.error_type,
                    "confidence": v.confidence
                }
                for v in result.votes
            ]
        }
        
    except ImportError as e:
        return {
            "has_hallucination": False,
            "error_type": "Import Error",
            "root_cause": f"multi_judge module not found: {e}",
            "confidence": 0.0,
            "agreement": 0.0,
            "votes": []
        }
    except RuntimeError as e:
        # Ollama not running
        return {
            "has_hallucination": False,
            "error_type": "Ollama Error",
            "root_cause": str(e),
            "confidence": 0.0,
            "agreement": 0.0,
            "votes": []
        }
    except Exception as e:
        return {
            "has_hallucination": False,
            "error_type": "System Error",
            "root_cause": str(e),
            "confidence": 0.0,
            "agreement": 0.0,
            "votes": []
        }


def get_detailed_report(source_code: str, generated_doc: str) -> str:
    """
    Get a detailed human-readable hallucination detection report.
    
    Args:
        source_code: The original source code.
        generated_doc: The AI-generated documentation.
    
    Returns:
        Formatted string report with all judge votes and analysis.
    """
    try:
        from multi_judge import MultiJudgeHallucinationDetector
        
        detector = MultiJudgeHallucinationDetector(auto_pull_models=True)
        result = detector.detect(source_code, generated_doc)
        return detector.get_detailed_report(result)
        
    except Exception as e:
        return f"Error generating report: {e}"


# Demo function
if __name__ == "__main__":
    print("=" * 60)
    print("HALLUCINATION DETECTION TEST")
    print("=" * 60)
    
    test_code = '''def add_numbers(a, b):
    """Add two numbers."""
    return a + b'''
    
    test_doc = "Adds two numbers together and returns the sum."
    bad_doc = "Multiplies two numbers and saves the result to a file."
    
    print("\n📋 Test Code:")
    print(test_code)
    
    print("\n✅ Testing good documentation...")
    result1 = detect_hallucination(test_code, test_doc)
    print(f"Result: {'FAIL' if result1['has_hallucination'] else 'PASS'}")
    print(f"Confidence: {result1.get('confidence', 0):.1%}")
    
    print("\n❌ Testing bad documentation...")
    result2 = detect_hallucination(test_code, bad_doc)
    print(f"Result: {'FAIL' if result2['has_hallucination'] else 'PASS'}")
    print(f"Error Type: {result2['error_type']}")
    print(f"Root Cause: {result2['root_cause']}")