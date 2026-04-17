from app.generation.llm import get_llm
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever


class RAGChain:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = get_llm()

    def run(self, query: str) -> str:
        # 🔥 Improve query for table retrieval
        enhanced_query = f"{query} capital expenditure breakdown table capex details"

        # =========================
        # STEP 1: RETRIEVE MORE DOCS
        # =========================
        docs = self.retriever.retrieve(enhanced_query, top_k=10)

        # =========================
        # STEP 2: RERANK (TABLE FOCUS)
        # =========================
        reranked_docs = self.reranker.rerank(enhanced_query, docs, top_k=4)

        # =========================
        # STEP 3: CONTEXT (KEEP RAW STRUCTURE)
        # =========================
        context = "\n\n".join(reranked_docs)

        # =========================
        # STEP 4: STRONG TABLE PROMPT 🔥
        # =========================
        prompt = f"""
You are a financial data extraction assistant.

Task:
Extract the COMPLETE capital expenditure breakdown table for the given mine.

Instructions:
- Return the FULL table (all rows and columns)
- Preserve structure (row names, values, units)
- If multiple tables exist, choose the one related to the mine
- Do NOT summarize
- Do NOT skip rows
- If table not found, say: "Table not found"

Output Format (STRICT):
Return as a clean markdown table.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()
