import os
import pickle
import tempfile

import faiss
import fitz
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(
    page_title="Financial Report AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Report AI")
st.caption("Upload a financial report and ask questions about its contents.")

# Load the models once per app process
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

embedding_model = load_models()

# Keep each browser session's document data separate
if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload a PDF financial report",
    type=["pdf"]
)


def extract_chunks(pdf_bytes, chunk_size=1000, overlap=200):
    """Extract text and create overlapping chunks with page numbers."""
    chunks = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_number
                    })

                if end == len(text):
                    break

                start += chunk_size - overlap
    finally:
        pdf.close()

    return chunks


def build_index(chunks):
    """Build a FAISS index for this uploaded PDF."""
    texts = [chunk["text"] for chunk in chunks]

    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def retrieve_chunks(question, document_data, top_k=3):
    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    count = min(top_k, document_data["index"].ntotal)
    scores, indices = document_data["index"].search(query_vector, count)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            **document_data["chunks"][idx],
            "score": float(score)
        })

    return results


def generate_answer(question, passages):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Configure it in .env."
        )

    context = "\n\n".join(
        f"Page {p['page']}:\n{p['text']}"
        for p in passages
    )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a financial report analysis assistant.

Answer the user's question using the supplied report passages.
Do not invent facts or numbers. If the evidence is insufficient,
say so. Preserve distinctions between standalone and consolidated
financial statements. Cite the supplied page numbers.

REPORT PASSAGES:
{context}

QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text or "No answer was returned."


# Process a newly uploaded document
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    # Rebuild only when the uploaded file changes
    import hashlib
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    current_data = st.session_state.document_data

    if current_data is None or current_data["file_hash"] != file_hash:
        with st.spinner("Extracting, chunking and indexing your report..."):
            chunks = extract_chunks(file_bytes)

            if not chunks:
                st.session_state.document_data = None
                st.error(
                    "No selectable text was found. This may be a scanned PDF "
                    "that needs OCR."
                )
                st.stop()

            index = build_index(chunks)

            st.session_state.document_data = {
                "file_hash": file_hash,
                "filename": uploaded_file.name,
                "chunks": chunks,
                "index": index
            }
            st.session_state.messages = []

        st.success("Report indexed successfully!")

    st.caption(
        f"Current report: {st.session_state.document_data['filename']} · "
        f"{len(st.session_state.document_data['chunks'])} chunks"
    )

    if st.button("Clear current report"):
        st.session_state.document_data = None
        st.session_state.messages = []
        st.rerun()

else:
    st.info("Upload a PDF to start asking questions.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View source passages"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}**")
                    st.write(source["text"][:1000])

# Chat input
question = st.chat_input(
    "Ask a question about your report...",
    disabled=st.session_state.document_data is None
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the report and generating an answer..."):
                passages = retrieve_chunks(
                    question,
                    st.session_state.document_data
                )

                if not passages:
                    answer = "I couldn't find relevant passages in this report."
                else:
                    answer = generate_answer(question, passages)

            st.markdown(answer)

            with st.expander("View source passages"):
                for source in passages:
                    st.markdown(f"**Page {source['page']}**")
                    st.write(source["text"][:1000])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": passages
            })

        except Exception as error:
            st.error(f"Something went wrong: {error}")