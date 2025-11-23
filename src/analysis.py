import os
from dotenv import load_dotenv

load_dotenv()

# Load Keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BYTEZ_KEY = os.getenv("BYTEZ_KEY")

use_bytez = BYTEZ_KEY is not None and BYTEZ_KEY.strip() != ""

if use_bytez:
    from bytez import Bytez
else:
    import openai
    # Set OpenAI key if not using Bytez
    if OPENAI_KEY:
        openai.api_key = OPENAI_KEY

def detect_hallucination(source_code, generated_doc, model_name="gpt-4"):
    """
    Uses a strong LLM to critique the generated documentation.
    Returns a structured dictionary: {Hallucination: bool, Error_Type: str, Reason: str}
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
            # Ensure we are asking for a specific model format expected by Bytez
            # Assuming 'openai/gpt-4' or similar strong model is available via Bytez
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
        
        # Parse the structured response
        lines = content.split('\n')
        for line in lines:
            if "Error Type:" in line:
                # Extract text after colon
                error_type = line.split(":", 1)[1].strip()
                # Fix common LLM output quirk where it writes "None" string
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