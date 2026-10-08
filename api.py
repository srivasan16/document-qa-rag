import os
from pathlib import Path

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "paragraph_chunks"
MODEL_NAME = "gemini-3.5-flash"
TOP_K = 3


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Set it in PowerShell before starting the API."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# CHROMADB CONFIGURATION
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Document QA API",
    description="RAG-based Document Question Answering API",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# RETRIEVE DOCUMENT CHUNKS
# ============================================================

def retrieve_chunks(question: str, top_k: int = TOP_K):

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question: str):

    documents, metadatas = retrieve_chunks(question)

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        document_name = metadata.get(
            "document",
            metadata.get("source", "Unknown document")
        )

        page_number = metadata.get(
            "page",
            metadata.get("page_number", "Unknown page")
        )

        context_parts.append(
            f"""
PASSAGE {i}
Document: {document_name}
Page: {page_number}

{document}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question ONLY using the provided document passages.

Rules:
1. Do not use outside knowledge.
2. Do not make up facts.
3. If the answer cannot be found in the passages, respond exactly:
   I don't know based on the provided documents.
4. For factual answers, include the document name and page number.
5. Include the relevant evidence passage.
6. Keep the answer clear and concise.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}

Return the answer in this format:

Answer:
<answer>

Source:
<document name>, Page <page number>

Evidence:
<relevant passage>
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )

    return response.text


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Document QA API is running",
        "status": "OK"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "model": MODEL_NAME
    }


# ============================================================
# QUESTION ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:
        answer = generate_answer(request.question)

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR IN /ask")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        print("=" * 70)

        return {
            "error": type(e).__name__,
            "message": str(e)
        }


