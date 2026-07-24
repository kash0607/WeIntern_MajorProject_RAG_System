from src.ingestion.pdf_loader import PDFLoader
from src.indexing.chunking import Chunking
from src.indexing.embeddings import EmbeddingModel
from src.indexing.vector_store import VectorStore

# -------------------------------
# Load PDF Documents
# -------------------------------
loader = PDFLoader("data/pdfs")
documents = loader.load_documents()

print("=" * 50)
print(f"Total Pages Loaded: {len(documents)}")
print("=" * 50)

# -------------------------------
# Chunk Documents
# -------------------------------
chunker = Chunking()

fixed_chunks = chunker.fixed_chunking(documents)
recursive_chunks = chunker.recursive_chunking(documents)

print("\nExample Fixed Chunk:\n")
print(fixed_chunks[0].page_content[:400])

print("\nExample Recursive Chunk:\n")
print(recursive_chunks[0].page_content[:400])

# -------------------------------
# Generate Embeddings
# -------------------------------
embedding_model = EmbeddingModel()

embeddings = embedding_model.generate_embeddings(recursive_chunks)

print(f"\nGenerated {len(embeddings)} embeddings")

# -------------------------------
# Store in ChromaDB
# -------------------------------
vector_db = VectorStore()

vector_db.store(recursive_chunks, embeddings)

print("\n✅ All documents successfully stored in ChromaDB!")