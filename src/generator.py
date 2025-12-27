import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# Load API Keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BYTEZ_KEY = os.getenv("BYTEZ_KEY")

# Import libraries conditionally
use_bytez = BYTEZ_KEY is not None and BYTEZ_KEY.strip() != ""

# Optional imports; guarded at use time
try:
    from bytez import Bytez
except Exception:
    Bytez = None

try:
    import openai
    if OPENAI_KEY:
        openai.api_key = OPENAI_KEY
except Exception:
    openai = None

try:
    import ollama
except Exception:
    ollama = None

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


def generate_documentation(
    code_snippet: str,
    language: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    provider: Optional[str] = None,
):
    """
    Generates documentation for a given code snippet.
    Provider options: "Bytez", "OpenAI", "Ollama", "Hugging Face".
    """
    system_prompt = (
        f"You are an expert senior developer. Your task is to write a clear, "
        f"concise {language} docstring explaining what this code does. Return ONLY the docstring."
        f"no code blocks, no backticks, no function definitions. 2 to 5 sentences explaining main functionality and exception handling."
        f"Use variable names surrounded by ` where appropriate."

    )
    user_prompt = f"Code:\n```{language}\n{code_snippet}\n```\n\nDocstring:"

    try:
        # Decide provider if not explicitly set
        if provider is None:
            if use_bytez and Bytez is not None:
                provider = "Bytez"
            elif OPENAI_KEY and openai is not None:
                provider = "OpenAI"
            elif ollama is not None:
                provider = "Ollama"
            else:
                provider = "Hugging Face"

        # Bytez
        if provider == "Bytez":
            if Bytez is None:
                return "Error: Bytez SDK not available"
            sdk = Bytez(BYTEZ_KEY)
            model = sdk.model(f"openai/{model_name}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            output, error, _ = model.run(messages)
            if error:
                return f"Error: {str(error)}"
            return output.get("content", "").strip()

        # OpenAI
        if provider == "OpenAI":
            if openai is None or not OPENAI_KEY:
                return "Error: OpenAI not configured"
            if hasattr(openai, 'chat'):
                client = openai.OpenAI(api_key=OPENAI_KEY)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message['content'].strip()

        # Ollama (local)
        if provider == "Ollama":
            if ollama is None:
                return "Error: Ollama python package not available"
            try:
                resp = ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": temperature},
                )
                # Python SDK returns dict with 'message': {'role','content'}
                msg = resp.get('message') or {}
                content = msg.get('content') or resp.get('response') or ""
                return str(content).strip()
            except Exception as e:
                return f"Error (Ollama): {str(e)}"

        # Hugging Face (Inference API)
        if provider == "Hugging Face":
            if InferenceClient is None:
                return "Error: huggingface_hub not available"
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            try:
                client = InferenceClient(model=model_name, token=hf_token)
                prompt = f"{system_prompt}\n\n{user_prompt}"
                # text_generation returns a string in recent versions
                output = client.text_generation(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=200,
                )
                return str(output).strip()
            except Exception as e:
                return f"Error (HF): {str(e)}"

        return "Error: Unknown provider"

    except Exception as e:
        return f"Error: {str(e)}"
