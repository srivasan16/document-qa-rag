import json
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load evaluation questions
# --------------------------------------------------

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

fixed_collection = client.get_collection(
    "fixed_size_chunks"
)

paragraph_collection = client.get_collection(
    "paragraph_chunks"
)


# --------------------------------------------------
# 4. Retrieval
# --------------------------------------------------

def retrieve(collection, question, top_k=3):

    embedding = model.encode(question).tolist()

    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )


# --------------------------------------------------
# 5. Calculate metrics
# --------------------------------------------------

def calculate_metrics(results, expected_pages):

    retrieved_pages = [
        metadata["page"]
        for metadata in results["metadatas"][0]
    ]

    # Recall@1
    recall_at_1 = (
        retrieved_pages[0] in expected_pages
    )

    # Recall@3
    recall_at_3 = any(
        page in expected_pages
        for page in retrieved_pages
    )

    # MRR
    reciprocal_rank = 0.0

    for rank, page in enumerate(
        retrieved_pages,
        start=1
    ):
        if page in expected_pages:
            reciprocal_rank = 1 / rank
            break

    return {
        "retrieved_pages": retrieved_pages,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "reciprocal_rank": reciprocal_rank
    }


# --------------------------------------------------
# 6. Run evaluation
# --------------------------------------------------

results = []

for item in questions:

    question_id = item["id"]
    question = item["question"]
    expected_pages = item["expected_pages"]

    print("\n" + "=" * 80)
    print(f"QUESTION {question_id}")
    print("=" * 80)
    print(question)

    print("Expected pages:", expected_pages)

    # Fixed-size
    fixed_results = retrieve(
        fixed_collection,
        question
    )

    # Paragraph
    paragraph_results = retrieve(
        paragraph_collection,
        question
    )

    # Metrics
    fixed_metrics = calculate_metrics(
        fixed_results,
        expected_pages
    )

    paragraph_metrics = calculate_metrics(
        paragraph_results,
        expected_pages
    )

    print("\nFixed-size pages:")
    print(fixed_metrics["retrieved_pages"])

    print(
        "Recall@1:",
        fixed_metrics["recall_at_1"]
    )

    print(
        "Recall@3:",
        fixed_metrics["recall_at_3"]
    )

    print(
        "MRR:",
        fixed_metrics["reciprocal_rank"]
    )

    print("\nParagraph pages:")
    print(paragraph_metrics["retrieved_pages"])

    print(
        "Recall@1:",
        paragraph_metrics["recall_at_1"]
    )

    print(
        "Recall@3:",
        paragraph_metrics["recall_at_3"]
    )

    print(
        "MRR:",
        paragraph_metrics["reciprocal_rank"]
    )

    results.append({
        "id": question_id,
        "question": question,
        "expected_pages": expected_pages,
        "fixed_size": fixed_metrics,
        "paragraph": paragraph_metrics
    })


# --------------------------------------------------
# 7. Calculate overall scores
# --------------------------------------------------

fixed_recall_1 = sum(
    r["fixed_size"]["recall_at_1"]
    for r in results
) / len(results)

fixed_recall_3 = sum(
    r["fixed_size"]["recall_at_3"]
    for r in results
) / len(results)

fixed_mrr = sum(
    r["fixed_size"]["reciprocal_rank"]
    for r in results
) / len(results)


paragraph_recall_1 = sum(
    r["paragraph"]["recall_at_1"]
    for r in results
) / len(results)

paragraph_recall_3 = sum(
    r["paragraph"]["recall_at_3"]
    for r in results
) / len(results)

paragraph_mrr = sum(
    r["paragraph"]["reciprocal_rank"]
    for r in results
) / len(results)


# --------------------------------------------------
# 8. Print final comparison
# --------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL CHUNKING COMPARISON")
print("=" * 80)

print("\nFIXED-SIZE")
print(
    f"Recall@1: {fixed_recall_1:.2%}"
)
print(
    f"Recall@3: {fixed_recall_3:.2%}"
)
print(
    f"MRR:      {fixed_mrr:.3f}"
)

print("\nPARAGRAPH")
print(
    f"Recall@1: {paragraph_recall_1:.2%}"
)
print(
    f"Recall@3: {paragraph_recall_3:.2%}"
)
print(
    f"MRR:      {paragraph_mrr:.3f}"
)


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

output = {
    "questions": results,
    "overall": {
        "fixed_size": {
            "recall_at_1": fixed_recall_1,
            "recall_at_3": fixed_recall_3,
            "mrr": fixed_mrr
        },
        "paragraph": {
            "recall_at_1": paragraph_recall_1,
            "recall_at_3": paragraph_recall_3,
            "mrr": paragraph_mrr
        }
    }
}


with open(
    "tests/retrieval_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("STEP 5 COMPLETE")
print(
    "Results saved to: "
    "tests/retrieval_results.json"
)