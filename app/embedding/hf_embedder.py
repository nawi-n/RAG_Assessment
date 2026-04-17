# app/embedding/hf_embedder.py

from typing import List

from sentence_transformers import SentenceTransformer

from app.logger import get_logger

logger = get_logger()


class HFEmbedder:
    def __init__(self):
        logger.info("Loading local embedding model...")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        embeddings = self.model.encode(
            texts,
            batch_size=8,  # 🔥 critical
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 🔥 improves retrieval quality
        )

        return embeddings.tolist()
