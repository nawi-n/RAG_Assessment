import os
from typing import List

import chromadb

from app.logger import get_logger

logger = get_logger()


class ChromaStore:
    def __init__(self, collection_name: str = "rag_collection"):
        logger.info("Initializing Chroma DB")

        os.makedirs("data/chroma", exist_ok=True)

        # ✅ Correct client (v1.x)
        self.client = chromadb.PersistentClient(path="data/chroma")

        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict] = None,
    ):
        logger.info(f"Storing {len(documents)} documents in Chroma")

        self.collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

        logger.info("Storage complete")
