import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self):

        print("Loading Retriever...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path="vectordb"
        )

        self.collection = self.client.get_collection(
            "rag_documents"
        )

    def create_query_embedding(self, query):

        return self.model.encode(query).tolist()

    def naive_search(self, query, top_k=5):

        query_embedding = self.create_query_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents","metadatas","distances"]
        )

        return results