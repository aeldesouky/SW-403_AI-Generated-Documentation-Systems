"""
Multi-Judge Hallucination Detection Module
==========================================

This module implements ensemble hallucination detection using 3 local Ollama models:
1. Code Searcher (qwen2.5-coder) - Finds code discrepancies
2. Reasoner (deepseek-r1) - Step-by-step logical analysis  
3. Thinker (gemma3) - Holistic pattern recognition

Each model votes on whether documentation contains hallucinations,
and the final verdict is determined by majority voting.

Author: AI Documentation Systems Team
"""

import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Try to import ollama Python library
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama Python library not installed. Run: pip install ollama")


class JudgeRole(Enum):
    """Specialized roles for each judge model."""
    CODE_SEARCHER = "code_searcher"
    REASONER = "reasoner"
    THINKER = "thinker"


class HallucinationType(Enum):
    """Types of hallucinations that can be detected."""
    NONE = "None"
    FABRICATED_VARIABLE = "Fabricated Variable"
    WRONG_LOGIC = "Wrong Logic"
    OMISSION = "Omission"
    IRRELEVANT = "Irrelevant"
    FABRICATED_FUNCTION = "Fabricated Function"
    WRONG_RETURN_TYPE = "Wrong Return Type"
    INCORRECT_PARAMETER = "Incorrect Parameter"


@dataclass
class JudgeVote:
    """Result from a single judge model."""
    model: str
    role: JudgeRole
    has_hallucination: bool
    error_type: str
    root_cause: str
    confidence: float  # 0.0 to 1.0
    raw_response: str = ""
    execution_time: float = 0.0


@dataclass
class EnsembleResult:
    """Aggregated result from all judge models."""
    final_verdict: bool  # True = has hallucination
    confidence: float    # Agreement level (0.0 to 1.0)
    error_type: str      # Most common error type
    root_cause: str      # Summary of causes
    votes: List[JudgeVote] = field(default_factory=list)
    agreement_ratio: float = 0.0
    total_execution_time: float = 0.0


# Model configuration for 16GB RAM
MODEL_CONFIG = {
    JudgeRole.CODE_SEARCHER: {
        "model": "qwen2.5-coder:7b",
        "description": "Analyzes code structure and finds discrepancies",
        "prompt_template": """You are a code analysis expert. Your job is to verify if the documentation accurately describes the code.

SOURCE CODE:
```
{source_code}
```

GENERATED DOCUMENTATION:
{generated_doc}

TASK: Search the code and verify each claim in the documentation.

For each claim:
1. Does this variable/function/parameter actually exist in the code?
2. Does the code actually perform the operations described?
3. Are there fabricated elements mentioned that don't exist?

Respond in this EXACT format:
Status: [PASS / FAIL]
Error Type: [None / Fabricated Variable / Wrong Logic / Omission / Irrelevant / Fabricated Function / Wrong Return Type / Incorrect Parameter]
Root Cause: [One sentence explanation]
Confidence: [0.0 to 1.0]"""
    },
    
    JudgeRole.REASONER: {
        "model": "deepseek-r1:7b",
        "description": "Step-by-step logical reasoning about claims",
        "prompt_template": """You are a logical reasoning expert analyzing documentation accuracy.

SOURCE CODE:
```
{source_code}
```

GENERATED DOCUMENTATION:
{generated_doc}

TASK: Analyze each documentation claim step-by-step.

Step 1 - EXTRACT: List each claim made in the documentation
Step 2 - VERIFY: For each claim, find supporting evidence in the code
Step 3 - EVALUATE: Is the claim fully supported, partially supported, or false?
Step 4 - CONCLUDE: Determine if there are hallucinations

Respond in this EXACT format:
Status: [PASS / FAIL]
Error Type: [None / Fabricated Variable / Wrong Logic / Omission / Irrelevant / Fabricated Function / Wrong Return Type / Incorrect Parameter]
Root Cause: [One sentence explanation]
Confidence: [0.0 to 1.0]"""
    },
    
    JudgeRole.THINKER: {
        "model": "gemma3:4b",
        "description": "Holistic pattern analysis and big-picture thinking",
        "prompt_template": """You are a documentation quality expert performing holistic analysis.

SOURCE CODE:
```
{source_code}
```

GENERATED DOCUMENTATION:
{generated_doc}

TASK: Analyze the documentation as a whole.

Consider:
1. Does the overall description match the code's actual purpose?
2. Are there any contradictions between the code and documentation?
3. Is anything critical missing from the documentation?
4. Does the documentation describe functionality that doesn't exist?

Respond in this EXACT format:
Status: [PASS / FAIL]  
Error Type: [None / Fabricated Variable / Wrong Logic / Omission / Irrelevant / Fabricated Function / Wrong Return Type / Incorrect Parameter]
Root Cause: [One sentence explanation]
Confidence: [0.0 to 1.0]"""
    }
}


class MultiJudgeHallucinationDetector:
    """
    Ensemble hallucination detection using multiple local Ollama models.
    
    Uses 3 specialized models that each analyze code-documentation pairs
    and vote on whether hallucinations are present.
    """
    
    def __init__(self, auto_pull_models: bool = False):
        """
        Initialize the multi-judge detector.
        
        Args:
            auto_pull_models: If True, automatically pull missing models (disabled by default to avoid hangs).
        """
        self.models = MODEL_CONFIG
        self.auto_pull = auto_pull_models
        self._available_models = []
        self._check_ollama()
        self._detect_available_models()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running and accessible."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama Python library not installed. Run: pip install ollama")
        
        try:
            ollama.list()
            return True
        except Exception as e:
            raise RuntimeError(f"Ollama is not running. Start it with: ollama serve\nError: {e}")
    
    def _detect_available_models(self):
        """Detect which models are available locally."""
        try:
            response = ollama.list()
            # Handle both dict and pydantic object responses
            if hasattr(response, 'models'):
                models_list = response.models
            else:
                models_list = response.get('models', [])
            
            # Get model names
            available_names = []
            for m in models_list:
                if hasattr(m, 'model'):
                    available_names.append(m.model)
                elif isinstance(m, dict) and 'name' in m:
                    available_names.append(m['name'])
            
            self._available_models = []
            for role, config in self.models.items():
                model_name = config['model']
                model_base = model_name.split(':')[0]
                if any(model_base in m for m in available_names):
                    self._available_models.append(role)
            
            if self._available_models:
                print(f"Available models: {[r.value for r in self._available_models]}")
            else:
                print("Warning: No judge models available. Run: ollama pull gemma3:4b")
        except Exception as e:
            print(f"Error detecting models: {e}")
    
    def get_available_roles(self) -> List[JudgeRole]:
        """Return list of roles with available models."""
        return self._available_models
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        try:
            response = ollama.list()
            if hasattr(response, 'models'):
                models_list = response.models
            else:
                models_list = response.get('models', [])
            
            available_names = []
            for m in models_list:
                if hasattr(m, 'model'):
                    available_names.append(m.model)
                elif isinstance(m, dict) and 'name' in m:
                    available_names.append(m['name'])
            
            model_base = model_name.split(':')[0]
            return any(model_base in m for m in available_names)
        except:
            return False
    
    def _call_model(self, role: JudgeRole, source_code: str, generated_doc: str) -> JudgeVote:
        """Call a single judge model and get its vote."""
        config = self.models[role]
        model_name = config["model"]
        
        # Build prompt
        prompt = config["prompt_template"].format(
            source_code=source_code,
            generated_doc=generated_doc
        )
        
        start_time = time.time()
        
        try:
            # Check if model is available (skip if not)
            if not self._is_model_available(model_name):
                return None  # Skip unavailable models
            
            # Call Ollama
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent analysis
                    "num_predict": 500   # Limit response length
                }
            )
            
            execution_time = time.time() - start_time
            raw_response = response.get('response', '')
            
            # Parse response
            return self._parse_response(raw_response, model_name, role, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return JudgeVote(
                model=model_name,
                role=role,
                has_hallucination=False,
                error_type="Error",
                root_cause=str(e),
                confidence=0.0,
                raw_response="",
                execution_time=execution_time
            )
    
    def _parse_response(self, response: str, model: str, role: JudgeRole, exec_time: float) -> JudgeVote:
        """Parse LLM response into structured JudgeVote."""
        # Default values
        has_hallucination = False
        error_type = "No Error"
        root_cause = "No issues found"
        confidence = 0.5
        
        # Parse Status
        if "Status: FAIL" in response or "Status:FAIL" in response:
            has_hallucination = True
        elif "Status: PASS" in response or "Status:PASS" in response:
            has_hallucination = False
        
        # Parse Error Type
        for line in response.split('\n'):
            if "Error Type:" in line:
                error_type = line.split(":", 1)[1].strip()
                if error_type.lower() == "none":
                    error_type = "No Error"
            elif "Root Cause:" in line:
                root_cause = line.split(":", 1)[1].strip()
            elif "Confidence:" in line:
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5
        
        return JudgeVote(
            model=model,
            role=role,
            has_hallucination=has_hallucination,
            error_type=error_type,
            root_cause=root_cause,
            confidence=confidence,
            raw_response=response,
            execution_time=exec_time
        )
    
    def detect(self, source_code: str, generated_doc: str, 
               parallel: bool = True) -> EnsembleResult:
        """
        Run ensemble hallucination detection.
        
        Args:
            source_code: The original source code.
            generated_doc: The generated documentation to verify.
            parallel: If True, run models in parallel (faster).
        
        Returns:
            EnsembleResult with aggregated verdict and individual votes.
        """
        start_time = time.time()
        votes: List[JudgeVote] = []
        
        # Only use available models
        roles_to_use = self._available_models if self._available_models else []
        
        if not roles_to_use:
            return EnsembleResult(
                final_verdict=False,
                confidence=0.0,
                error_type="No Models",
                root_cause="No judge models available. Run: ollama pull gemma3:4b",
                votes=[],
                agreement_ratio=0.0,
                total_execution_time=0.0
            )
        
        if parallel and len(roles_to_use) > 1:
            # Run available models in parallel
            with ThreadPoolExecutor(max_workers=len(roles_to_use)) as executor:
                futures = {
                    executor.submit(self._call_model, role, source_code, generated_doc): role
                    for role in roles_to_use
                }
                
                for future in as_completed(futures):
                    try:
                        vote = future.result()
                        if vote is not None:  # Skip None results from unavailable models
                            votes.append(vote)
                    except Exception as e:
                        print(f"Error in model execution: {e}")
        else:
            # Run sequentially
            for role in roles_to_use:
                vote = self._call_model(role, source_code, generated_doc)
                if vote is not None:
                    votes.append(vote)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        return self._aggregate_votes(votes, total_time)
    
    def _aggregate_votes(self, votes: List[JudgeVote], total_time: float) -> EnsembleResult:
        """Aggregate individual votes into final verdict."""
        if not votes:
            return EnsembleResult(
                final_verdict=False,
                confidence=0.0,
                error_type="No Votes",
                root_cause="No models returned results",
                votes=[],
                agreement_ratio=0.0,
                total_execution_time=total_time
            )
        
        # Count votes
        hallucination_votes = sum(1 for v in votes if v.has_hallucination)
        pass_votes = len(votes) - hallucination_votes
        
        # Majority voting
        final_verdict = hallucination_votes > pass_votes
        
        # Calculate agreement ratio
        majority_count = max(hallucination_votes, pass_votes)
        agreement_ratio = majority_count / len(votes)
        
        # Calculate weighted confidence
        if final_verdict:
            # Average confidence of models that detected hallucination
            fail_votes = [v for v in votes if v.has_hallucination]
            avg_confidence = sum(v.confidence for v in fail_votes) / len(fail_votes) if fail_votes else 0.5
        else:
            # Average confidence of models that passed
            pass_votes_list = [v for v in votes if not v.has_hallucination]
            avg_confidence = sum(v.confidence for v in pass_votes_list) / len(pass_votes_list) if pass_votes_list else 0.5
        
        # Adjust confidence based on agreement
        final_confidence = avg_confidence * agreement_ratio
        
        # Determine most common error type
        error_types = [v.error_type for v in votes if v.error_type not in ["No Error", "Error", "Model Unavailable"]]
        if error_types:
            from collections import Counter
            error_type = Counter(error_types).most_common(1)[0][0]
        else:
            error_type = "No Error" if not final_verdict else "Unknown"
        
        # Compile root causes
        causes = [v.root_cause for v in votes if v.root_cause and v.root_cause != "No issues found"]
        root_cause = "; ".join(causes[:3]) if causes else "No specific issues identified"
        
        return EnsembleResult(
            final_verdict=final_verdict,
            confidence=final_confidence,
            error_type=error_type,
            root_cause=root_cause,
            votes=votes,
            agreement_ratio=agreement_ratio,
            total_execution_time=total_time
        )
    
    def get_detailed_report(self, result: EnsembleResult) -> str:
        """Generate a detailed human-readable report from ensemble result."""
        lines = [
            "=" * 60,
            "MULTI-JUDGE HALLUCINATION DETECTION REPORT",
            "=" * 60,
            "",
            f"FINAL VERDICT: {'❌ HALLUCINATION DETECTED' if result.final_verdict else '✅ PASS - No Hallucination'}",
            f"Confidence: {result.confidence:.1%}",
            f"Agreement: {result.agreement_ratio:.1%} ({sum(1 for v in result.votes if v.has_hallucination == result.final_verdict)}/{len(result.votes)} models)",
            f"Error Type: {result.error_type}",
            f"Root Cause: {result.root_cause}",
            f"Total Time: {result.total_execution_time:.2f}s",
            "",
            "-" * 60,
            "INDIVIDUAL JUDGE VOTES:",
            "-" * 60,
        ]
        
        for vote in result.votes:
            status = "FAIL" if vote.has_hallucination else "PASS"
            lines.extend([
                f"",
                f"  [{vote.role.value.upper()}] {vote.model}",
                f"    Status: {status} | Confidence: {vote.confidence:.1%}",
                f"    Error Type: {vote.error_type}",
                f"    Reason: {vote.root_cause}",
                f"    Time: {vote.execution_time:.2f}s",
            ])
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Compatibility function for existing analysis.py
def detect_hallucination(source_code: str, generated_doc: str, model_name: str = None) -> Dict[str, Any]:
    """
    Drop-in replacement for the original detect_hallucination function.
    Uses multi-judge ensemble instead of single GPT-4 call.
    
    Args:
        source_code: The source code being documented.
        generated_doc: The AI-generated documentation.
        model_name: Ignored (kept for backward compatibility).
    
    Returns:
        Dictionary with has_hallucination, error_type, and root_cause.
    """
    try:
        detector = MultiJudgeHallucinationDetector(auto_pull_models=False)
        result = detector.detect(source_code, generated_doc, parallel=False)
        
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
                    "confidence": v.confidence
                }
                for v in result.votes
            ]
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


def demo():
    """Demonstrate the multi-judge hallucination detector."""
    print("=" * 60)
    print("MULTI-JUDGE HALLUCINATION DETECTOR DEMO")
    print("=" * 60)
    
    # Sample code
    test_code = '''def calculate_discount(price, percentage):
    """Calculate discounted price."""
    if percentage < 0 or percentage > 100:
        raise ValueError("Invalid percentage")
    return price * (1 - percentage / 100)'''
    
    # Good documentation (should PASS)
    good_doc = "Calculates the discounted price by applying a percentage discount. Raises ValueError if percentage is not between 0 and 100."
    
    # Hallucinated documentation (should FAIL)
    bad_doc = "Calculates the discounted price and logs the result to a database. Returns a tuple of (original_price, discounted_price, tax_amount)."
    
    print("\n📋 Test Code:")
    print(test_code)
    
    try:
        detector = MultiJudgeHallucinationDetector(auto_pull_models=True)
        
        print("\n" + "=" * 60)
        print("TEST 1: Good Documentation (should PASS)")
        print("=" * 60)
        print(f"Doc: {good_doc}")
        result1 = detector.detect(test_code, good_doc)
        print(detector.get_detailed_report(result1))
        
        print("\n" + "=" * 60)
        print("TEST 2: Hallucinated Documentation (should FAIL)")
        print("=" * 60)
        print(f"Doc: {bad_doc}")
        result2 = detector.detect(test_code, bad_doc)
        print(detector.get_detailed_report(result2))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print("2. Models are available: ollama pull qwen2.5-coder:7b deepseek-r1:7b gemma3:4b")


if __name__ == "__main__":
    demo()
