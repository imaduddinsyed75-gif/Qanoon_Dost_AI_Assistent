"""Session-only PDF upload and retrieval for Phase 4.

This module deliberately keeps uploaded-document chunks and embeddings in
memory. It never writes to, or imports data into, the main legal knowledge
base files.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from llm_response import (
    build_context,
    build_prompt,
    call_gemini,
    call_groq,
)

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
MAX_FILE_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class UploadedDocument:
    """In-memory representation of one uploaded PDF."""

    name: str
    chunks: tuple[dict, ...]
    embeddings: np.ndarray


def load_embedding_model() -> SentenceTransformer:
    """Load the Phase 2-compatible embedding model."""

    return SentenceTransformer(MODEL_NAME)


def extract_pdf_pages(file: bytes | BinaryIO) -> list[dict]:
    """Extract text while retaining the source page for citations."""

    if hasattr(file, "read"):
        data = file.read()
    else:
        data = file
    if not isinstance(data, bytes) or not data:
        raise ValueError("The uploaded PDF is empty or could not be read.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Please upload a PDF smaller than 15 MB.")

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text:
                pages.append({"page": page_number, "text": text})

    if not pages:
        raise ValueError(
            "No selectable text was found. This PDF may be scanned; "
            "please upload a text-based PDF."
        )
    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_document(
    name: str,
    pdf_file: bytes | BinaryIO,
    model: SentenceTransformer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> UploadedDocument:
    """Extract, chunk, and embed a PDF into an in-memory document."""

    if not name.strip():
        raise ValueError("The uploaded document needs a file name.")
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    for page in extract_pdf_pages(pdf_file):
        for text in _split_text(page["text"], chunk_size, overlap):
            chunks.append(
                {
                    "text": text,
                    "law_name": name,
                    "category": "User uploaded document",
                    "jurisdiction": "User document",
                    "year": "N/A",
                    "page_start": page["page"],
                    "page_end": page["page"],
                }
            )

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return UploadedDocument(name, tuple(chunks), np.asarray(embeddings))


def retrieve_document(
    document: UploadedDocument,
    query: str,
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Retrieve the most relevant chunks from one uploaded document."""

    if not query or not query.strip():
        raise ValueError("Question must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    query_embedding = model.encode(
        [query.strip()], normalize_embeddings=True, show_progress_bar=False
    )[0]
    scores = document.embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][: min(top_k, len(document.chunks))]

    results = []
    for index in indices:
        result = dict(document.chunks[int(index)])
        result["score"] = round(float(scores[int(index)]), 4)
        results.append(result)
    return results


def get_document_answer(
    document: UploadedDocument,
    question: str,
    model: SentenceTransformer,
    language: str = "roman_urdu",
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Retrieve context from an uploaded PDF and ask the configured LLM."""

    results = retrieve_document(document, question, model, top_k)
    context = build_context(results)
    prompt = build_prompt(question, context, language)

    try:
        answer = call_gemini(prompt)
        provider = "gemini"
    except Exception as gemini_error:
        try:
            answer = call_groq(prompt)
            provider = "groq"
        except Exception as groq_error:
            raise RuntimeError(
                f"Both providers failed. Gemini error: {gemini_error}; "
                f"Groq error: {groq_error}"
            ) from groq_error

    return {
        "answer": answer,
        "sources": results,
        "provider_used": provider,
        "question": question,
        "document_name": document.name,
    }
