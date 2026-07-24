from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def generate_embeddings(self, chunks):

        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        return embeddings