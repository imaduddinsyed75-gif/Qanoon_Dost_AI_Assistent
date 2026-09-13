import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"

TOP_K = 8


print("Loading data...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(EMBEDDINGS_FILE)

if len(chunks) != len(embeddings):
    raise ValueError(
        f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
    )


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


query = input("\nAsk a legal question: ").strip()

if not query:
    raise ValueError("Question cannot be empty.")


query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)[0]


# Cosine similarity because stored/query embeddings are normalized
scores = embeddings @ query_embedding

top_indices = np.argsort(scores)[::-1][:TOP_K]


print("\nTop results:\n")


for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    page_start = chunk.get(
        "page_start",
        chunk.get("page")
    )

    page_end = chunk.get(
        "page_end",
        page_start
    )

    if page_start == page_end:
        page_display = str(page_start)
    else:
        page_display = f"{page_start}-{page_end}"

    print("=" * 70)
    print("Rank:", rank)
    print("Score:", round(float(scores[index]), 4))
    print("Law:", chunk["law_name"])
    print("Category:", chunk["category"])
    print("Jurisdiction:", chunk["jurisdiction"])
    print("Year:", chunk["year"])
    print("Page(s):", page_display)

    print()
    print(chunk["text"])
    print()