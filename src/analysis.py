"""
Hallucination Detection Module
==============================

This module provides hallucination detection for AI-generated documentation.
Supports both local Ollama models and cloud APIs (OpenAI/Bytez) as fallback.

Author: AI Documentation Systems Team
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

# Load Keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BYTEZ_KEY = os.getenv("BYTEZ_KEY")

use_bytez = BYTEZ_KEY is not None and BYTEZ_KEY.strip() != ""

if use_bytez:
    from bytez import Bytez
else:
    import openai
    if OPENAI_KEY:
        openai.api_key = OPENAI_KEY


def detect_hallucination(source_code: str, generated_doc: str, 
                         model_name: str = "gpt-4") -> Dict[str, Any]:
    """
    Detect hallucinations in AI-generated documentation.
    
    Uses local Ollama models first (if available), falls back to cloud APIs.
    
    Args:
        source_code: The original source code being documented.
        generated_doc: The AI-generated documentation to verify.
        model_name: Model to use for cloud API fallback (default: gpt-4).
    
    Returns:
        Dictionary containing:
        - has_hallucination: bool - True if hallucination detected
        - error_type: str - Type of error (Fabricated Variable, Wrong Logic, etc.)
        - root_cause: str - Explanation of the issue
    """
    # Try local Ollama models first
    try:
        from multi_judge import MultiJudgeHallucinationDetector
        
        detector = MultiJudgeHallucinationDetector(auto_pull_models=False)
        
        # Check if any models are available
        if detector.get_available_roles():
            result = detector.detect(source_code, generated_doc, parallel=False)
            return {
                "has_hallucination": result.final_verdict,
                "error_type": result.error_type,
                "root_cause": result.root_cause
            }
    except (ImportError, RuntimeError):
        pass  # Fall through to cloud API
    
    # Fallback to cloud API (OpenAI or Bytez)
    return _detect_with_cloud_api(source_code, generated_doc, model_name)


def _detect_with_cloud_api(source_code: str, generated_doc: str, 
                           model_name: str = "gpt-4") -> Dict[str, Any]:
    """
    Uses cloud LLM (OpenAI/Bytez) to critique the generated documentation.
    Returns a structured dictionary: {has_hallucination: bool, error_type: str, root_cause: str}
    """
    
    judge_prompt = f"""
    You are a QA auditor for software documentation.
    
    Source Code:
    {source_code}
    
    Generated Documentation:
    {generated_doc}
    
    Task: Determine if the documentation contains 'Hallucinations' (claims not supported by the code) or 'Omissions' (critical info missing).
    
    Return your analysis in this EXACT format:
    Status: [PASS / FAIL]
    Error Type: [None / Fabricated Variable / Wrong Logic / Omission / Irrelevant]
    Root Cause: [One sentence explanation]
    """

    messages = [{"role": "user", "content": judge_prompt}]

    try:
        content = ""
        
        # --- PATH A: BYTEZ ---
        if use_bytez:
            sdk = Bytez(BYTEZ_KEY)
            model_id = f"openai/{model_name}" 
            model = sdk.model(model_id)
            
            output, error, _ = model.run(messages)
            
            if error:
                return {"has_hallucination": False, "error_type": "API Error", "root_cause": str(error)}
            
            content = output.get("content", "").strip()

        # --- PATH B: OPENAI (Official) ---
        else:
            if hasattr(openai, 'chat'):
                client = openai.OpenAI(api_key=OPENAI_KEY)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0
                )
                content = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0
                )
                content = response.choices[0].message['content']

        # --- PARSING LOGIC ---
        is_fail = "Status: FAIL" in content
        error_type = "No Error"
        cause = "No error"
        
        lines = content.split('\n')
        for line in lines:
            if "Error Type:" in line:
                error_type = line.split(":", 1)[1].strip()
                if error_type.lower() == "none": 
                    error_type = "No Error"
            if "Root Cause:" in line:
                cause = line.split(":", 1)[1].strip()
                
        return {
            "has_hallucination": is_fail,
            "error_type": error_type,
            "root_cause": cause
        }

    except Exception as e:
        return {"has_hallucination": False, "error_type": "Crash", "root_cause": str(e)}