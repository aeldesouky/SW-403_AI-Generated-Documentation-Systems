import os
from dotenv import load_dotenv

load_dotenv()

# Load API Keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BYTEZ_KEY = os.getenv("BYTEZ_KEY")

# Import libraries conditionally
use_bytez = BYTEZ_KEY is not None and BYTEZ_KEY.strip() != ""

if use_bytez:
    from bytez import Bytez
else:
    import openai
    openai.api_key = OPENAI_KEY


def generate_documentation(code_snippet, language, model_name="gpt-3.5-turbo", temperature=0.2):
    """
    Generates documentation for a given code snippet.
    """
    system_prompt = f"You are an expert senior developer. Your task is to write a clear, concise {language} docstring explaining what this code does. Return ONLY the docstring."
    
    user_prompt = f"Code:\n```{language}\n{code_snippet}\n```\n\nDocstring:"

    try:
        # Use Bytez if available
        if use_bytez:
            sdk = Bytez(BYTEZ_KEY)
            model = sdk.model(f"openai/{model_name}")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Bytez does NOT support a temperature parameter
            output, error, _ = model.run(messages)

            if error:
                return f"Error: {str(error)}"

            return output.get("content", "").strip()

        # Otherwise use OpenAI (new or old style)
        if hasattr(openai, 'chat'): 
            # For newer library versions
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        else:
            # Fallback for older library versions (0.28)
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
            
    except Exception as e:
        return f"Error: {str(e)}"
