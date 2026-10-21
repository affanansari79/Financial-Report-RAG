import os
import pickle
import faiss

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Load the FAISS index
index = faiss.read_index("financial_report.index")

# Load chunk text and page metadata
with open("chunk_metadata.pkl", "rb") as file:
    chunks = pickle.load(file)

# Load the same embedding model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_chunks(question, top_k=3):
    """Retrieve the most relevant passages from the report."""

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    passages = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[idx]

        passages.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })

    return passages


def generate_answer(question, passages):
    """Generate an answer using only the retrieved report passages."""

    if not passages:
        return "I couldn't find relevant passages in the report."

    context = "\n\n".join(
        f"Page {p['page']}:\n{p['text']}"
        for p in passages
    )

    prompt = f"""
You are a financial report analysis assistant.

Answer the user's question using the report passages below.

Rules:
1. Use the supplied passages as your evidence.
2. Do not invent facts or numbers.
3. If the passages do not contain enough information,
   clearly say so.
4. Preserve distinctions between standalone and consolidated
   financial statements.
5. Distinguish revenue from revenue from operations.
6. Include the relevant report page number in your answer.
7. Explain calculations if you perform any.

REPORT PASSAGES:
{context}

USER QUESTION:
{question}

Write a clear, direct answer.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text or "Gemini returned an empty response."


print("Financial Report Q&A")
print("Type 'exit' to quit.")

while True:
    question = input("\nAsk a question: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    print("\nSearching your annual report...")

    passages = retrieve_chunks(question)

    print("\nGenerating answer with Gemini...\n")

    try:
        answer = generate_answer(question, passages)
        print("ANSWER:")
        print(answer)

        print("\nSOURCE PASSAGES:")
        for p in passages:
            print(
                f"\nPage {p['page']} "
                f"(similarity: {p['score']:.3f})"
            )
            print(p["text"][:500])

    except Exception as e:
        print(f"Error generating answer: {e}")