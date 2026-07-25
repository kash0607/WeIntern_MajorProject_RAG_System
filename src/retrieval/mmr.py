import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


class MMRRetriever:

    def __init__(self):

        print("Loading MMR Retriever...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path="vectordb"
        )

        self.collection = self.client.get_collection(
            "rag_documents"
        )

    def cosine_similarity(self, a, b):

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

    def search(self, query, top_k=5, fetch_k=20, lambda_param=0.7):

        query_embedding = self.model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_k,
            include=[
                "documents",
                "metadatas",
                "embeddings",
                "distances"
            ]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        embeddings = np.array(results["embeddings"][0])

        selected = []
        selected_indices = []

        # Pick the most relevant document first
        selected.append(0)
        selected_indices.append(0)

        while len(selected) < top_k:

            best_score = -1
            best_index = None

            for i in range(len(embeddings)):

                if i in selected_indices:
                    continue

                relevance = self.cosine_similarity(
                    query_embedding,
                    embeddings[i]
                )

                diversity = max(
                    self.cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )
                    for j in selected_indices
                )

                mmr_score = (
                    lambda_param * relevance
                    - (1 - lambda_param) * diversity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = i

            selected.append(best_index)
            selected_indices.append(best_index)

        final_results = []

        for idx in selected_indices:

            final_results.append({
                "document": documents[idx],
                "metadata": metadatas[idx]
            })

        return final_results