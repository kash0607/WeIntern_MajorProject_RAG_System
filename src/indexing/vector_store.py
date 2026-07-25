import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="vectordb"
        )

        self.collection = self.client.get_or_create_collection(
            name="rag_documents"
        )

    def store(self, chunks, embeddings):

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):

            ids.append(str(i))
            documents.append(chunk.page_content)

            metadata = {
                "page": chunk.metadata.get("page", "Unknown"),
                "source": chunk.metadata.get("source", "Unknown")
            }

            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(f"{len(ids)} vectors stored successfully.")