from src.retrieval.mmr import MMRRetriever

retriever = MMRRetriever()

results = retriever.search(
    "What is Deep Learning?"
)

for i, result in enumerate(results, start=1):

    print("=" * 80)
    print(f"Result {i}")
    print(f"Page : {result['metadata']['page']}")
    print("-" * 80)
    print(result["document"][:400])
    print()