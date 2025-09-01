import google.generativeai as genai
import os

class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env or environment.")
        genai.configure(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        model = genai.GenerativeModel(self.model)
        prompt = f"System instructions:\n{system}\n\nUser request:\n{user}"
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        return resp.text
