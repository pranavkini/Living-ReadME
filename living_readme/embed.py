import google.generativeai as genai

class Embedder:
    def __init__(self, api_key: str, model: str = "models/embedding-001"):
        genai.configure(api_key=api_key)
        self.model = model
        self.llm_model = "gemini-2.5-flash"

    def embed(self, text: str):
        """Generate embeddings for text."""
        result = genai.embed_content(model=self.model, content=text)
        return result["embedding"]

    def llm_answer(self, question: str, context: str):
        """Ask Gemini with retrieved context."""
        prompt = f"""You are an assistant that answers questions about a code repository.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""
        response = genai.GenerativeModel(self.llm_model).generate_content(prompt)
        return response.text.strip()
