import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"

print("Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]

print("Total chunks:", len(texts))

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save(EMBEDDINGS_FILE, embeddings)

print("Embeddings created successfully.")
print("Shape:", embeddings.shape)
print("Saved to:", EMBEDDINGS_FILE)