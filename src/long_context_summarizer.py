# src/long_context_summarizer.py
import os
from google import genai


def summarize_long_context(text, model_name):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Map friendly names → actual Gemini model IDs
        GEMINI_MODEL_MAP = {
            "gemini-2.5-pro": "models/gemini-2.5-pro",
            "gemini-2.5-flash": "models/gemini-2.5-flash",
            "gemini-pro-latest": "models/gemini-pro-latest",
        }

        model_id = GEMINI_MODEL_MAP.get(model_name)
        if not model_id:
            return f"Error: Unsupported Gemini model {model_name}"

        response = client.models.generate_content(
            model=model_id,
            contents=f"""
You are a biomedical research assistant.
Summarize the following Alzheimer’s clinical trial text.
Focus on:
- Drug names
- Trial phase
- Mechanism of action
- Outcomes
- Limitations

TEXT:
{text}
"""
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"