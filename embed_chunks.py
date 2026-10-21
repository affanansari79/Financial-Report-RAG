import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the chunks created in Step 2
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No chunks found. Check chunks.json first.")

texts = [chunk["text"] for chunk in chunks]

print("Loading embedding model...")

# A compact model suitable for a first RAG project
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype="float32")

# Create a FAISS index for similarity search
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save the index and corresponding chunk metadata
faiss.write_index(index, "financial_report.index")

with open("chunk_metadata.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("\nEmbedding completed!")
print("Total chunks indexed:", index.ntotal)
print("Embedding dimensions:", embeddings.shape[1])
print("Saved: financial_report.index")
print("Saved: chunk_metadata.pkl")