from chromadb import PersistentClient

class RAGPipeline:
    def __init__(self, embedder, persist_directory="./chroma"):
        self.embedder = embedder
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("repo_docs")

    def query(self, question: str, n_results: int = 3) -> str:
        # Embed the question
        q_emb = self.embedder.embed(question)

        # Retrieve nearest docs
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
        )

        # Get contexts
        contexts = results.get("documents", [[]])[0]
        context_text = "\n\n".join(contexts) if contexts else "No relevant context found."

        # Generate answer with Gemini
        answer = self.embedder.llm_answer(question, context_text)
        return answer
