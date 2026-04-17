from typing import List

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        # ⚡ use base for speed
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, docs: List[str], top_k: int = 5) -> List[str]:
        if not docs:
            return []

        pairs = [(query, doc) for doc in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]
