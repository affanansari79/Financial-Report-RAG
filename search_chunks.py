import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the saved FAISS index
index = faiss.read_index("financial_report.index")

# Load the original chunks and page numbers
with open("chunk_metadata.pkl", "rb") as file:
    chunks = pickle.load(file)

# Load the SAME embedding model used earlier
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Financial Report Search")
print("Type 'exit' to quit.")

while True:
    question = input("\nAsk a question: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    # Convert the question into an embedding
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve the 3 most similar chunks
    scores, indices = index.search(query_vector, 3)

    print("\nMost relevant passages:\n")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx < 0:
            continue

        chunk = chunks[idx]

        print(f"--- Result {rank} ---")
        print(f"Page: {chunk['page']}")
        print(f"Similarity score: {score:.3f}")
        print(chunk["text"])
        print()