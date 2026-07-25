from src.retrieval.retriever import Retriever

print("=" * 50)
print("Production RAG System")
print("=" * 50)

retriever = Retriever()

while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    results = retriever.naive_search(question)
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for i, (doc, meta, score) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print("=" * 80)
        print(f"Result {i}")
        print(f"Score : {score:.4f}")
        print(f"Source: {meta.get('source', 'Unknown')}")
        print(f"Page  : {meta.get('page', 'Unknown')}")
        print("-" * 80)
        print(doc[:500])
        print()

        