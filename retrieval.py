import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 8

# Small ranking bonus when the user explicitly mentions
# a province/jurisdiction.
JURISDICTION_BONUS = 0.10


# --------------------------------------------------
# Load knowledge base
# --------------------------------------------------

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find chunks file: {CHUNKS_FILE}"
    )

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find embeddings file: {EMBEDDINGS_FILE}"
    )


with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(EMBEDDINGS_FILE)


if len(chunks) != len(embeddings):
    raise ValueError(
        f"Mismatch: {len(chunks)} chunks but "
        f"{len(embeddings)} embeddings. "
        "Run build_embeddings.py again."
    )


# --------------------------------------------------
# Load embedding model once
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Jurisdiction detection
# --------------------------------------------------

def detect_jurisdiction(query):
    """
    Detect whether the user explicitly mentions a jurisdiction.

    Returns:
        "Punjab"
        "Sindh"
        "Khyber Pakhtunkhwa"
        "Federal"
        or None
    """

    query = query.lower().strip()

    patterns = {
        "Khyber Pakhtunkhwa": [
            r"\bkhyber\s+pakhtunkhwa\b",
            r"\bkpk\b",
            r"\bkp\b",
        ],

        "Sindh": [
            r"\bsindh\b",
            r"\bsind\b",
        ],

        "Punjab": [
            r"\bpunjab\b",
        ],

        "Federal": [
            r"\bfederal\b",
            r"\bislamabad\b",
            r"\bict\b",
            r"\bislamabad capital territory\b",
        ],
    }

    for jurisdiction, jurisdiction_patterns in patterns.items():
        for pattern in jurisdiction_patterns:
            if re.search(pattern, query):
                return jurisdiction

    return None


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve(query, top_k=DEFAULT_TOP_K):
    """
    Retrieve the most relevant legal chunks for a user question.

    Retrieval uses:
    1. SentenceTransformer semantic similarity
    2. A small jurisdiction bonus when the question explicitly
       mentions Punjab, Sindh, Khyber Pakhtunkhwa, or Federal/ICT.

    Args:
        query (str):
            User's legal question.

        top_k (int):
            Number of results to return.

    Returns:
        List of dictionaries containing:
        - text
        - law_name
        - category
        - jurisdiction
        - year
        - page_start
        - page_end
        - score
    """

    # ------------------------------
    # Validate query
    # ------------------------------

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query = query.strip()

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    top_k = min(top_k, len(chunks))


    # ------------------------------
    # Create query embedding
    # ------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]


    # ------------------------------
    # Semantic similarity
    # ------------------------------

    # Document embeddings were also normalized when created,
    # so dot product here is cosine similarity.
    semantic_scores = embeddings @ query_embedding


    # Make a copy because jurisdiction boosting should not
    # modify the original semantic scores.
    ranking_scores = semantic_scores.copy()


    # ------------------------------
    # Jurisdiction-aware ranking
    # ------------------------------

    detected_jurisdiction = detect_jurisdiction(query)

    if detected_jurisdiction:
        for index, chunk in enumerate(chunks):

            if chunk.get("jurisdiction") == detected_jurisdiction:
                ranking_scores[index] += JURISDICTION_BONUS


    # ------------------------------
    # Select best chunks
    # ------------------------------

    top_indices = np.argsort(
        ranking_scores
    )[::-1][:top_k]


    # ------------------------------
    # Build results
    # ------------------------------

    results = []

    for index in top_indices:

        chunk = chunks[index]

        page_start = chunk.get(
            "page_start",
            chunk.get("page")
        )

        page_end = chunk.get(
            "page_end",
            page_start
        )

        results.append({
            "text": chunk["text"],
            "law_name": chunk["law_name"],
            "category": chunk["category"],
            "jurisdiction": chunk["jurisdiction"],
            "year": chunk["year"],
            "page_start": page_start,
            "page_end": page_end,

            # Keep score as the true semantic similarity,
            # not the artificial jurisdiction-boosted score.
            "score": round(
                float(semantic_scores[index]),
                4
            )
        })

    return results