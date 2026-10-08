import os
from pathlib import Path

import chromadb
from google import genai
from google.genai import types


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "paragraph_chunks"

MODEL_NAME = "gemini-3.5-flash"
TOP_K = 3


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Set it in PowerShell before running the program."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# CHROMADB CONFIGURATION
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

try:
    collection = chroma_client.get_collection(
        COLLECTION_NAME
    )
except Exception as e:
    raise RuntimeError(
        f"Could not find ChromaDB collection '{COLLECTION_NAME}'.\n"
        f"Make sure Step 4 was completed successfully.\n\n"
        f"Original error: {e}"
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, top_k=TOP_K):
    """
    Retrieve the most relevant chunks from ChromaDB.
    """

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents, metadatas):
    """
    Convert retrieved chunks into a source-grounded context.
    """

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        document_name = metadata.get(
            "document",
            "Unknown document"
        )

        page_number = metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
SOURCE {i}

Document: {document_name}
Page: {page_number}

Passage:
{document}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    # --------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # --------------------------------------------------------

    documents, metadatas, distances = retrieve_chunks(
        question,
        TOP_K
    )

    # --------------------------------------------------------
    # Step 2: Check whether anything was retrieved
    # --------------------------------------------------------

    if not documents:
        return (
            "I don't know based on the provided documents."
        )

    # --------------------------------------------------------
    # Step 3: Build context
    # --------------------------------------------------------

    context = build_context(
        documents,
        metadatas
    )

    # --------------------------------------------------------
    # Step 4: Create strict grounding prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question ONLY using
the retrieved passages provided below.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT make up facts.
3. If the retrieved passages do not contain enough
   information to answer the question, respond exactly:

I don't know based on the provided documents.

4. Keep the answer concise and clear.
5. Every factual answer MUST include a citation.
6. The citation must contain:
   - Document name
   - Page number
7. Include the most relevant evidence passage.
8. Do not cite information that is not present in
   the retrieved passages.

USER QUESTION:
{question}

RETRIEVED SOURCES:
{context}

REQUIRED ANSWER FORMAT:

Answer:
<concise answer>

Source:
<Document name>, Page <page number>

Evidence:
<short relevant passage from the retrieved source>
"""

    # --------------------------------------------------------
    # Step 5: Call Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )

    # --------------------------------------------------------
    # Step 6: Return generated answer
    # --------------------------------------------------------

    if not response.text:
        return (
            "I don't know based on the provided documents."
        )

    return response.text


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Government Report Intelligence - Document Q&A")
    print("=" * 70)

    print(f"\nVector database : {CHROMA_PATH}")
    print(f"Collection      : {COLLECTION_NAME}")
    print(f"Model           : {MODEL_NAME}")
    print(f"Top-K           : {TOP_K}")

    print("\nSystem ready.")

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print(
            "\nI don't know based on the provided documents."
        )

    else:

        try:

            answer = generate_answer(question)

            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70)

            print(answer)

            print("=" * 70)

        except Exception as e:

            print("\nERROR")
            print("=" * 70)
            print(str(e))
            print("=" * 70)