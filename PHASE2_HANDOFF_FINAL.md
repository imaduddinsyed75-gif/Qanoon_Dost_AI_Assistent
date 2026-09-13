# RightsGuide AI — Phase 1 & Phase 2 Handoff

## Overview

This part of RightsGuide AI provides the legal knowledge base and retrieval pipeline used by the later LLM/RAG stages.

The current knowledge base contains Pakistani legal documents covering:

- Tenant law
- Consumer law
- Labor law
- Family law
- Criminal procedure

The retrieval system accepts a user's legal question and returns the most semantically relevant legal passages together with source metadata.

## Legal Documents

The current knowledge base contains 9 legal documents:

1. Punjab Rented Premises Act 2009
2. Punjab Consumer Protection Act 2005
3. Punjab Minimum Wages Act 2019
4. Punjab Shops and Establishments Ordinance 1969
5. Muslim Family Laws Ordinance 1961
6. Code of Criminal Procedure 1898

7. West Pakistan Urban Rent Restriction Ordinance 1959 — Khyber Pakhtunkhwa — tenant

8. Sind Rented Premises Ordinance 1979 — Sindh — tenant

9. Payment of Wages Act 1936 — Federal — labor

## Pipeline

The Phase 1 + Phase 2 pipeline is:

PDF legal documents  
→ text extraction  
→ text cleaning  
→ legal-aware chunking  
→ SentenceTransformer embeddings  
→ NumPy cosine-similarity retrieval  
→ top relevant legal passages

## Embedding Model

Model: `all-MiniLM-L6-v2`

Library: `sentence-transformers`

Embedding dimension: `384`

## Current Knowledge Base

Final number of chunks: `1189`

Important files:

- `chunks.json` — legal text chunks and metadata
- `embeddings.npy` — precomputed normalized embeddings
- `metadata.json` — metadata for the source legal documents
- `retrieval.py` — reusable retrieval interface
- `test_retrieval.py` — retrieval testing script
- `search_laws.py` — optional interactive retrieval/testing script
- `build_embeddings.py` — creates embeddings from `chunks.json`
- `chunk_documents.py` — creates legal-aware chunks from processed documents

## Retrieval Method

The main legal knowledge base does NOT require FAISS or ChromaDB.

Retrieval uses:

- SentenceTransformer embeddings
- NumPy
- Cosine similarity

The embeddings are generated with `all-MiniLM-L6-v2`.

The stored document embeddings and query embeddings are normalized. Because the embeddings are normalized, NumPy dot-product similarity represents cosine similarity.

## Using Retrieval in Phase 3

The main interface for Phase 3 is `retrieval.py`.

Import the retrieval function:

```python
from retrieval import retrieve
```

Then call it like this:

```python
results = retrieve(
    "My landlord is trying to evict me. What are my rights?",
    top_k=5
)
```

The function returns the most relevant legal chunks.

## Returned Result Format

Each result is a Python dictionary containing:

```python
{
    "text": "...",
    "law_name": "...",
    "category": "...",
    "jurisdiction": "...",
    "year": 2009,
    "page_start": 7,
    "page_end": 7,
    "score": 0.4714
}
```

Fields:

- `text` — retrieved legal passage
- `law_name` — name of the source law
- `category` — legal category
- `jurisdiction` — jurisdiction of the law
- `year` — year of the law
- `page_start` — first source PDF page represented by the chunk
- `page_end` — last source PDF page represented by the chunk
- `score` — semantic similarity score

## Example Phase 3 Integration

Phase 3 can use the retrieved passages as context for Gemini or Groq.

```python
from retrieval import retrieve

question = "My landlord is trying to evict me. What are my rights?"

results = retrieve(question, top_k=5)

context_parts = []

for result in results:
    if result["page_start"] == result["page_end"]:
        page_info = f'Page {result["page_start"]}'
    else:
        page_info = f'Pages {result["page_start"]}-{result["page_end"]}'

    source = f'{result["law_name"]}, {page_info}'

    context_parts.append(
        f'Source: {source}\n'
        f'{result["text"]}'
    )

context = "\n\n".join(context_parts)

print(context)
```

The Phase 3 LLM can then receive:

1. The user's legal question
2. The retrieved legal context
3. Instructions to answer using the supplied legal context
4. Instructions to provide the relevant law/source citation
5. Instructions to simplify the answer for the user

## Source Citations

Every retrieved result includes law name, jurisdiction, year, starting PDF page, and ending PDF page.

For example:

`Punjab Rented Premises Act 2009 — Page 7`

If a chunk spans multiple pages:

`Punjab Rented Premises Act 2009 — Pages 6-7`

## Testing

The retrieval pipeline has been tested using questions from all major categories in the current knowledge base.

### Tenant Law

Example question:

`My landlord is trying to evict me. What are my rights?`

Relevant provisions from the Punjab Rented Premises Act 2009 were successfully retrieved, including grounds for eviction.

### Consumer Law

Example question:

`I bought a defective product and the seller refuses to refund me. What rights do I have?`

Relevant provisions from the Punjab Consumer Protection Act 2005 were successfully retrieved, including defective-product liability, Consumer Court procedure, replacement/refund remedies, and return/refund policy.

### Labor Law

Example question:

`My employer is paying me less than the minimum wage. What can I do?`

Relevant provisions from the Punjab Minimum Wages Act 2019 were successfully retrieved, including claims arising from wages paid below the minimum wage.

### Family Law

Example question:

`Can a husband marry another woman without permission while his first marriage still exists?`

Relevant provisions from the Muslim Family Laws Ordinance 1961 were successfully retrieved, including Section 6 relating to polygamy and permission of the Arbitration Council.

### Criminal Procedure

Example question:

`Can police arrest a person without a warrant, and when are they allowed to do so?`

Relevant provisions from the Code of Criminal Procedure 1898 were successfully retrieved, including provisions relating to arrest without warrant and subsequent procedure.

## Testing Retrieval

To quickly test the reusable retrieval function, run:

```powershell
python test_retrieval.py
```

The test question can be changed inside `test_retrieval.py`.

## Important for Phase 3

Phase 3 should normally use:

```python
from retrieval import retrieve
```

There is no need for Phase 3 to:

- rebuild the legal documents
- rerun PDF extraction
- regenerate chunks
- regenerate embeddings
- use ChromaDB
- use FAISS

The precomputed legal knowledge base can be loaded directly by `retrieval.py`.

## Important for Phase 4 — User PDF Upload

Phase 4 may independently process user-uploaded PDFs.

The main legal knowledge base uses precomputed SentenceTransformer embeddings with NumPy cosine-similarity retrieval.

User-uploaded documents may use their own temporary/session-only vector store.

For compatibility, use the same embedding model where practical: `all-MiniLM-L6-v2`.

The uploaded-document pipeline may therefore look like:

User PDF  
→ document loader  
→ text extraction  
→ chunking  
→ `all-MiniLM-L6-v2` embeddings  
→ temporary/session vector store  
→ retrieval

This does NOT conflict with the main Phase 2 knowledge base.

Uploaded user documents should remain temporary/session-only and should not be permanently added to the main legal knowledge base.

## Rebuilding Embeddings

Do NOT rebuild `embeddings.npy` unless `chunks.json` changes.

Currently:

`chunks.json = 1189 chunks`

`embeddings.npy = 1189 x 384 embeddings`

These currently match.

If `chunks.json` is regenerated or modified, rebuild the embeddings by running:

```powershell
python build_embeddings.py
```

After rebuilding, the number of embeddings must match the number of chunks.

## File Responsibilities

### `retrieval.py`

Main reusable Phase 2 interface. Used by other phases to retrieve relevant legal context.

### `test_retrieval.py`

Testing script for verifying that `retrieval.py` works correctly. This file can be kept for development and integration testing.

### `search_laws.py`

Interactive/manual retrieval testing script. Useful for debugging but not required by Phase 3.

### `chunks.json`

Contains the final legal text chunks and metadata.

Current chunk count: `1189`

### `embeddings.npy`

Contains the precomputed normalized embeddings for all chunks.

Current shape: `(1189, 384)`

### `metadata.json`

Contains document-level information such as law name, category, jurisdiction, year, and processed file location.

### `chunk_documents.py`

Creates the legal-aware chunks from the processed legal documents. It preserves source page information and allows chunks to contain context spanning PDF page boundaries.

### `build_embeddings.py`

Generates embeddings for all chunks using `all-MiniLM-L6-v2`.

## Phase 2 Architecture Summary

```text
Pakistani Legal PDFs
        ↓
Extracted Text
        ↓
Cleaned Legal Text
        ↓
Legal-Aware Chunking
        ↓
chunks.json
        ↓
SentenceTransformer
all-MiniLM-L6-v2
        ↓
embeddings.npy
        ↓
User Question
        ↓
Query Embedding
        ↓
NumPy Cosine Similarity
        ↓
Top-K Legal Chunks
        ↓
retrieval.py
        ↓
Phase 3 LLM
Gemini / Groq
```

## Final Status

Phase 1 — Foundation / Legal Data Collection: **COMPLETE**

Phase 2 — RAG Retrieval Pipeline: **COMPLETE**

Current knowledge base:

- 9 legal documents
- 5 legal categories
- 1189 legal chunks
- 384-dimensional embeddings
- Semantic retrieval with NumPy cosine similarity

- Jurisdiction-aware ranking for explicit province/federal queries
- Source law metadata
- Source page metadata
- Cross-page citation support
- Reusable `retrieve()` function
- Retrieval tested across tenant, consumer, labor, family, and criminal questions

Phase 3 can integrate directly with `retrieval.py`.
