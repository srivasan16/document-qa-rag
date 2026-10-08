import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------
print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

# Correct collection names
fixed_collection = client.get_collection("fixed_size_chunks")
paragraph_collection = client.get_collection("paragraph_chunks")

# --------------------------------------------------
# 3. Test question
# --------------------------------------------------
question = "What is the role of NITI Aayog?"

print("\nQuestion:")
print(question)

# Convert question into embedding
query_embedding = model.encode(question).tolist()

# --------------------------------------------------
# 4. Search Fixed-Size collection
# --------------------------------------------------
fixed_results = fixed_collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# --------------------------------------------------
# 5. Search Paragraph collection
# --------------------------------------------------
paragraph_results = paragraph_collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# --------------------------------------------------
# 6. Display Fixed-Size results
# --------------------------------------------------
print("\n" + "=" * 70)
print("FIXED-SIZE CHUNK RESULTS")
print("=" * 70)

for i in range(3):

    metadata = fixed_results["metadatas"][0][i]

    print(f"\n--- Result {i + 1} ---")
    print("Document:", metadata["document"])
    print("Page:", metadata["page"])
    print("Distance:", fixed_results["distances"][0][i])
    print("\nEvidence:")
    print(fixed_results["documents"][0][i][:1000])

# --------------------------------------------------
# 7. Display Paragraph results
# --------------------------------------------------
print("\n" + "=" * 70)
print("PARAGRAPH CHUNK RESULTS")
print("=" * 70)

for i in range(3):

    metadata = paragraph_results["metadatas"][0][i]

    print(f"\n--- Result {i + 1} ---")
    print("Document:", metadata["document"])
    print("Page:", metadata["page"])
    print("Distance:", paragraph_results["distances"][0][i])
    print("\nEvidence:")
    print(paragraph_results["documents"][0][i][:1000])