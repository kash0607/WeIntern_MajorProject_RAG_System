from src.retrieval.retriever import Retriever
from src.llm.ollama_llm import OllamaLLM

retriever = Retriever()
llm = OllamaLLM()

print("=" * 50)
print("Production RAG System")
print("=" * 50)

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    results = retriever.naive_search(question)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)

    answer = llm.generate(
        question=question,
        context=context
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for meta in metadatas:
        print(
            f"{meta.get('source')} (Page {meta.get('page')})"
        )