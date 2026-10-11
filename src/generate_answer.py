import os
import time
from pathlib import Path
import time
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "paragraph_chunks"

MODEL_NAME = "gemini-3.8-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 2

# ============================================================
# GEMINI
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Please set your Gemini API key."
    )

client = genai.Client(api_key=api_key)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )

# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents, metadatas):

    parts = []

    for i, (doc, meta) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        parts.append(
            f"""
SOURCE {i}

Document: {meta.get("document")}

Page: {meta.get("page")}

Passage:
{doc}
"""
        )

    return "\n".join(parts)

# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    start = time.perf_counter()

    documents, metadatas, distances = retrieve_chunks(
        question,
        TOP_K
    )

    print("\n========== RETRIEVED DOCUMENTS ==========")

    for i, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        print(f"\nChunk {i}")
        print(f"Document : {meta['document']}")
        print(f"Page     : {meta['page']}")
        print(f"Distance : {dist:.4f}")
        print(doc[:500])
    
    print("\n=========================================")

    if len(documents) == 0:
        return "I don't know based on the provided documents."

    context = build_context(
        documents,
        metadatas
    )

    prompt = f"""
You are an AI assistant answering questions from documents.

Use ONLY the provided context.

If the answer exists anywhere in the context,
summarize it in your own words.

If the answer truly does not exist,
reply exactly:

I don't know based on the provided documents.

Context:

{context}

Question:

{question}

Answer:
"""

    response = None
    MAX_RETRIES = 5

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            
            print("=" * 80)
            print(context)
            print("=" * 80)

            end = time.perf_counter()

            print("=" * 60)
            print("Prompt Size:", len(prompt))
            print(f"Response Time: {end - start:.2f} seconds")
            print("=" * 60)
            print(response.text)

            return response.text.strip()

        except Exception as e:
            wait = 2 ** attempt
            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)

    if response is None:
        end = time.perf_counter()
        print(f"Response Time: {end - start:.2f} seconds")
        return "Gemini API temporarily unavailable."

    if response.text:
        end = time.perf_counter()
        print(f"Response Time: {end - start:.2f} seconds")
        return response.text.strip()

    end = time.perf_counter()
    print(f"Response Time: {end - start:.2f} seconds")
    return "I don't know based on the provided documents."

# ============================================================
# RAG RESPONSE
# ============================================================

def get_rag_response(question):

    documents, metadatas, distances = retrieve_chunks(
        question,
        TOP_K
    )

    answer = generate_answer(question)

    return {
        "question": question,
        "answer": answer,
        "contexts": documents,
        "metadata": metadatas,
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Government Report Intelligence - Document Q&A")
    print("=" * 70)

    print(f"\nVector Database : {CHROMA_PATH}")
    print(f"Collection      : {COLLECTION_NAME}")
    print(f"Model           : {MODEL_NAME}")
    print(f"Top-K           : {TOP_K}")

    print("\nSystem Ready!")

    while True:

        question = input(
            "\nEnter your question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = generate_answer(question)

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)
        print("=" * 70)

if __name__ == "__main__":
    main()