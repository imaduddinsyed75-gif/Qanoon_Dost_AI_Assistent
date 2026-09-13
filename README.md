# ⚖️ Qanoon Dost AI — Legal Rights Assistant for Pakistan

> **"Qanoon Dost"** means *Friend of the Law* in Urdu.  
> AI-powered legal guidance, grounded in real Pakistani legislation — in English, Urdu, or Roman Urdu.

---

## Problem Statement

Millions of Pakistanis face everyday legal problems — wrongful eviction, unpaid wages, consumer disputes, family law questions — but cannot afford a lawyer and cannot read dense legislative text. Legal aid is scarce, trust in informal advice is low, and the language barrier makes official documents inaccessible to most citizens.

**Qanoon Dost AI** bridges this gap: it reads the actual law, answers questions in plain language, and cites the exact source so citizens can verify and follow up with confidence.

---

## What It Does

| Feature | Description |
|---|---|
| 📚 RAG over Pakistani law | Retrieves the most relevant legal passages from a curated corpus using semantic search |
| 🤖 Dual-LLM pipeline | Gemini 2.0 Flash as primary; Groq (LLaMA 3.3 70B) as automatic fallback |
| 🌍 Multi-language answers | English, Urdu script, or Roman Urdu — user's choice |
| 📄 Upload your own PDF | Ask questions about a private contract, tenancy agreement, or notice |
| 🌐 Live Web Search (Phase 8) | Optional Tavily fallback when local knowledge base confidence is low |
| 💬 Persistent chat history | Conversations saved locally with SQLite |
| ⚖️ Source citations | Every answer lists the law, jurisdiction, and page range it relied on |

---

## Pakistani Law Coverage

| Domain | Legislation Covered |
|---|---|
| 🏠 **Tenant / Landlord** | Punjab Rented Premises Act 2009 · Sindh Rented Premises Ordinance 1979 · West Pakistan Urban Rent Restriction Ordinance 1959 |
| 🛒 **Consumer Rights** | Punjab Consumer Protection Act 2005 |
| 👷 **Labour / Wages** | Payment of Wages Act 1936 · Punjab Minimum Wages Act 2019 · Punjab Shops & Establishments Ordinance 1969 |
| 👨‍👩‍👧 **Family Law** | Muslim Family Laws Ordinance 1961 |
| ⚖️ **Criminal Procedure** | Code of Criminal Procedure 1898 |

Jurisdiction-aware ranking boosts results matching the province the user mentions (Punjab, Sindh, KPK, Federal / ICT).

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────┐
│         Streamlit UI  (app.py)      │
│  sidebar toggle: Live Web Search    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       LLM Response  (llm_response.py)│
│                                     │
│  1. retrieve() — semantic search    │
│     over embeddings.npy + chunks.json│
│     (SentenceTransformer MiniLM-L6) │
│                                     │
│  2a. confidence ≥ threshold?        │
│       → build_prompt() → Gemini     │
│                   └── fallback Groq │
│                                     │
│  2b. confidence < threshold         │
│      AND web search ON?             │
│       → Tavily search               │
│       → build_web_prompt() → Gemini │
│                   └── fallback Groq │
└──────────────┬──────────────────────┘
               │
               ▼
         Structured response
         { answer, sources, provider_used, web_search_used }
```

### Component Map

| File | Role |
|---|---|
| `app.py` | Streamlit UI, session state, chat history, PDF upload tab |
| `llm_response.py` | Prompt building, Gemini/Groq calls, Tavily web search fallback |
| `retrieval.py` | Semantic retrieval with jurisdiction-aware re-ranking |
| `chunks.json` | Pre-processed legal text chunks with metadata |
| `embeddings.npy` | Pre-computed MiniLM-L6-v2 embeddings for all chunks |
| `user_documents.py` | In-memory PDF indexing for user-uploaded documents |
| `chat_storage.py` | SQLite-backed conversation persistence |

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/qanoon-dost-ai.git
cd qanoon-dost-ai
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Copy the example secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Fill in your keys in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY  = "your_gemini_api_key_here"
GROQ_API_KEY    = "your_groq_api_key_here"
TAVILY_API_KEY  = "your_tavily_api_key_here"   # optional — needed for Live Web Search
```

> **Free-tier accounts:**  
> Gemini API key → [aistudio.google.com](https://aistudio.google.com)  
> Groq API key → [console.groq.com](https://console.groq.com)  
> Tavily API key → [app.tavily.com](https://app.tavily.com)

### 3. Run

```bash
streamlit run app.py
```

---

## Streamlit Community Cloud Deployment

1. Push the repository to GitHub (make sure `venv/` and `.env` are in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo and `app.py`.
3. Open **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY  = "your_gemini_api_key_here"
GROQ_API_KEY    = "your_groq_api_key_here"
TAVILY_API_KEY  = "your_tavily_api_key_here"
```

4. Click **Deploy**. No `.env` file is needed on Cloud — `st.secrets` is used automatically.

> ⚠️ `chunks.json` and `embeddings.npy` must be committed to the repository (they are pre-built, not generated at runtime). They are already included in the repo.

---

## Phase 8 — Live Web Search

When the **🌐 Live Web Search** toggle in the sidebar is **ON**:

- If the best local knowledge-base similarity score is below `0.30`, the system automatically falls back to a Tavily web search.
- The answer is generated from the top-3 web results and clearly marked:

> ⚠️ *This information is retrieved from the web, please verify.*

- `TAVILY_API_KEY` must be set in Streamlit Secrets (or `.streamlit/secrets.toml` locally) for this feature to work. If the key is missing, the system silently stays with the local knowledge base.

---

## Key Design Decisions

**Why RAG instead of fine-tuning?**  
Pakistani legislation is public but sparse in LLM training data. RAG grounds every answer in the actual text, prevents hallucination, and makes source citation trivial.

**Why Gemini + Groq fallback?**  
Gemini 2.0 Flash is fast and accurate; Groq provides near-instant inference on open-weight models. The dual setup ensures near-zero downtime.

**Why Streamlit Secrets over `.env`?**  
`.env` files cannot be used securely on Streamlit Community Cloud. All secrets go through `st.secrets`, which maps transparently to the Cloud dashboard and to a local `secrets.toml` — one code path works everywhere.

---

## Disclaimer

Qanoon Dost AI provides general legal information for educational and informational purposes only. It is not a substitute for advice from a licensed legal professional. Always consult a qualified lawyer for your specific situation.

---

*Built for the Pakistan AI Hackathon · September 2026*
