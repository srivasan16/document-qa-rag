"""
Step 4 — Embeddings + ChromaDB retrieval

Creates two separate ChromaDB collections:
- fixed_size_chunks
- paragraph_chunks

Both use the SAME embedding model so the chunking comparison is fair.

Default embedding model:
sentence-transformers/all-MiniLM-L6-v2

Run:
    python src/build_vector_db.py

Optional retrieval test:
    python src/build_vector_db.py --query "What are the objectives of NITI Aayog?"
"""

from pathlib import Path
import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
DB_DIR = BASE_DIR / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_collection(client, collection_name, records, model):
    # Re-running the script replaces the collection so results are reproducible.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        texts = [r["text"] for r in batch]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        ids = [r["chunk_id"] for r in batch]

        metadatas = [
            {
                "strategy": r["strategy"],
                "document": r["document"],
                "page": int(r["page"]),
                "word_count": int(r["word_count"])
            }
            for r in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return collection


def search(collection, model, query, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(result["ids"][0])):
        hits.append({
            "rank": i + 1,
            "chunk_id": result["ids"][0][i],
            "distance": result["distances"][0][i],
            "document": result["metadatas"][0][i]["document"],
            "page": result["metadatas"][0][i]["page"],
            "text": result["documents"][0][i],
        })

    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    fixed_records = load_jsonl(CHUNKS_DIR / "fixed_size_chunks.jsonl")
    paragraph_records = load_jsonl(CHUNKS_DIR / "paragraph_chunks.jsonl")

    fixed_collection = build_collection(
        client, "fixed_size_chunks", fixed_records, model
    )
    paragraph_collection = build_collection(
        client, "paragraph_chunks", paragraph_records, model
    )

    print()
    print("STEP 4 COMPLETE")
    print(f"Fixed-size collection: {fixed_collection.count()} chunks")
    print(f"Paragraph collection: {paragraph_collection.count()} chunks")
    print(f"Database: {DB_DIR}")

    if args.query:
        print()
        print("=" * 80)
        print(f"QUERY: {args.query}")
        print("=" * 80)

        for name, collection in [
            ("FIXED-SIZE", fixed_collection),
            ("PARAGRAPH-BASED", paragraph_collection),
        ]:
            print()
            print(f"--- {name} ---")
            for hit in search(collection, model, args.query, args.top_k):
                preview = " ".join(hit["text"].split())[:350]
                print(
                    f"[{hit['rank']}] "
                    f"{hit['document']} | page {hit['page']} | "
                    f"distance={hit['distance']:.4f}"
                )
                print(preview)
                print()


if __name__ == "__main__":
    main()
