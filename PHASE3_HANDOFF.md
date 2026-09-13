# RightsGuide AI — Phase 3 Handoff (LLM Integration)

## Status: COMPLETE

Phase 3 connects Phase 2's legal knowledge base (`retrieval.py`) to Gemini
(primary) and Groq (fallback) to produce simplified, source-grounded
legal answers.

## Main File

`llm_response.py` — import and use this:

```python
from llm_response import get_legal_answer

result = get_legal_answer(
    "My landlord is trying to evict me. What are my rights?",
    language="roman_urdu"   # or "english" / "urdu"
)

result["answer"]          # the AI's answer (string)
result["sources"]         # list of source law dicts
result["provider_used"]   # "gemini" or "groq"
```

## API Keys — Two Supported Methods

**Method 1 — Streamlit Secrets (used automatically once deployed)**
Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`,
fill in your keys. On Streamlit Community Cloud, paste the same into
the app's Secrets settings in the dashboard.

**Method 2 — .env / Colab Secrets (local testing)**
Copy `.env.example` → `.env`, fill in your keys.

The code checks Streamlit Secrets first, then falls back to
environment variables — no code changes needed between local
and deployed use.

⚠️ Never commit `.env` or `.streamlit/secrets.toml` — both are in
`.gitignore` already.

## New Packages (added to requirements.txt)

```
google-genai
groq
python-dotenv
streamlit
```

## Testing

```
python test_llm.py
```

Runs 3 sample questions (tenant, consumer, labor) through the full
pipeline and prints the answer, provider used, and sources.

## For Phase 4 (User Document Upload) — Next Step

Phase 4 should build a similar function for user-uploaded PDFs,
independent of the main knowledge base:

```
User PDF → extract text → chunk → embed (all-MiniLM-L6-v2)
→ temporary/session vector store → retrieve top chunks
→ same build_context() / build_prompt() style as llm_response.py
→ call_gemini() / call_groq() (can reuse these functions)
```

Reuse `call_gemini()`, `call_groq()`, `build_context()`, and
`build_prompt()` from `llm_response.py` where possible — don't
duplicate the API-calling logic.

Keep uploaded documents session-only. Do NOT add them to the main
`chunks.json` / `embeddings.npy` knowledge base.

## Phase 4 — Implemented

Phase 4 is available in `user_documents.py` and `app.py`.

- Upload text-based PDFs from the **Ask Your PDF** tab.
- Text is extracted with page numbers, split into overlapping chunks, and
  embedded with `all-MiniLM-L6-v2`.
- Chunks and embeddings stay in Streamlit session memory only.
- Answers use the same grounded prompt and Gemini → Groq fallback as Phase 3.
- The main `chunks.json` and `embeddings.npy` files are never modified.

Run the UI with:

```bash
streamlit run app.py
```

## For Phase 5 (UI) — What to Expect

`get_legal_answer()` returns a fixed dict shape:

```python
{
    "answer": str,
    "sources": [
        {"law_name": str, "jurisdiction": str, "year": int,
         "page_start": int, "page_end": int, "score": float},
        ...
    ],
    "provider_used": "gemini" | "groq",
    "question": str
}
```

UI can render `answer` in the chat bubble and `sources` in an
expandable "Sources" section underneath.
