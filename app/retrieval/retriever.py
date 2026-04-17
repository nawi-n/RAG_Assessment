from typing import List

from app.embedding.hf_embedder import HFEmbedder
from app.vectorstore.chroma_store import ChromaStore


class Retriever:
    def __init__(self):
        self.embedder = HFEmbedder()
        self.vector_store = ChromaStore()

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedder.embed([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results.get("documents", [[]])[0]
