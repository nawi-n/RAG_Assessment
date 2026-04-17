from app.generation.rag_chain import RAGChain

rag = RAGChain()

while True:
    q = input("\nQuery: ")
    if q == "exit":
        break

    print("\nAnswer:\n", rag.run(q))
