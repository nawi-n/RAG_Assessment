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
        You are a senior financial analyst specializing in mining company annual reports.

        Task:
        Extract CAPITAL EXPENDITURE (CAPEX) information for the specified mine using ONLY the provided context.

        IMPORTANT UNDERSTANDING:
        CAPEX may NOT be explicitly written as "CAPEX". It may appear as:
        - Capital expenditure
        - Capital spending
        - Investment in property, plant and equipment (PP&E)
        - Cash used in investing activities
        - Sustaining or growth capital investments

        Rules:
        1. Use ONLY the provided context. Do NOT use external knowledge.
        2. Do NOT assume or hallucinate missing values.
        3. CAPEX may appear in tables OR narrative text — both are valid.
        4. If multiple CAPEX-related values exist, include all of them clearly.
        5. If data is split across chunks, combine only if explicitly supported by context.
        6. If no CAPEX-related information is found, say: "CAPEX information not found in the provided context."

        Output Format:
        - If a table exists → present it as a clean markdown table
        - If only text exists → summarize clearly in bullet points
        - If both exist → show both (table first, then explanation)
        - Keep numbers exactly as in the context
        - Do not add interpretations beyond the data

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        response = self.llm.invoke(prompt)

        return response.content.strip()
