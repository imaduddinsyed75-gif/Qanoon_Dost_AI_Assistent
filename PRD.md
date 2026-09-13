# ⚖️ QANOON DOST AI
### Legal Rights Assistant for Citizens of Pakistan
**Product Requirements Document | AI Hackathon 2026**  
**Team Lead:** Hafiz Syed Imad ud Din | **Live App:** [law-dost-ai.streamlit.app](https://law-dost-ai.streamlit.app)

---

## 1. PROBLEM STATEMENT
Pakistan has over 220 million citizens, yet access to legal knowledge remains a privilege for a wealthy few. Millions of Pakistanis face everyday legal challenges — wrongful eviction by landlords, wage theft by employers, consumer fraud, and family disputes — but cannot afford a lawyer and cannot read dense legislative language.

Legal aid organizations are underfunded, courts are backlogged, and unofficial advice from family and social media is unreliable and often dangerous. The language barrier makes official legislation — written in formal English or Urdu legalese — inaccessible to the majority.

**Qanoon Dost AI** solves this by providing instant, source-grounded, plain-language legal guidance in English, Urdu, or Roman Urdu — for free, at scale.

---

## 2. SOLUTION OVERVIEW
Qanoon Dost AI ("Friend of the Law") is a Retrieval-Augmented Generation (RAG) system that reads the actual text of Pakistani legislation and answers citizen questions grounded in those exact legal sources — with citations.

- **Multilingual:** Natural language questions in English, Urdu, or Roman Urdu.
- **Source-grounded:** Cites exact law name, jurisdiction, and page number.
- **Dual-LLM:** Uses Gemini 2.0 Flash (primary) + Groq LLaMA 3.3 (fallback).
- **Document Upload:** Users can upload their own tenancy agreement or notice for analysis.
- **Web Search Fallback:** Live Tavily web search if local knowledge base confidence is low.
- **Deployed:** Deployed on Streamlit Community Cloud (free tier).

---

## 3. LEGAL KNOWLEDGE BASE
Pakistani statutory coverage across 5 key civil areas:
- 🏠 **Tenant / Landlord:** Punjab Rented Premises Act 2009, Sindh Rented Premises Ordinance 1979, West Pakistan Urban Rent Restriction Ordinance 1959.
- 🛒 **Consumer Rights:** Punjab Consumer Protection Act 2005.
- 👷 **Labour / Wages:** Payment of Wages Act 1936, Punjab Minimum Wages Act 2019, Punjab Shops & Establishments Ordinance 1969.
- 👨‍👩‍👧 **Family Law:** Muslim Family Laws Ordinance 1961.
- ⚖️ **Criminal Procedure:** Code of Criminal Procedure 1898.

---

## 4. SYSTEM ARCHITECTURE
The system follows a strict RAG pipeline to prevent hallucinations:
1. **Retrieval:** User question $\rightarrow$ SentenceTransformer `MiniLM-L6-v2` embedding $\rightarrow$ cosine similarity search over `embeddings.npy`.
2. **Re-ranking:** Top-k chunks with jurisdiction-aware filtering.
3. **Generation:** Grounded prompt construction $\rightarrow$ Gemini 2.0 Flash $\rightarrow$ Groq LLaMA 3.3 fallback.
4. **Web Fallback:** If similarity score $< 0.30$ and web toggle is ON $\rightarrow$ Tavily API retrieval.
5. **Response:** Answer + statutory citations rendered to Streamlit UI.

---

## 5. TEAM STRUCTURE
| Team Member | Role | Primary Module |
| :--- | :--- | :--- |
| **Imad (Team Lead)** | Lead Architect & Deployment | GitHub, Git, Cloud Deployment (Streamlit Cloud), System Architecture |
| **Mehak Imran** | Backend & RAG Engineer | RAG Pipeline, `llm_response.py`, Gemini Integration |
| **Saba Arshad** | Frontend & UX Engineer | Streamlit UI, Multilingual Chat (`app.py`), Presentation |
| **Muhammad Ammar Fakhar** | Legal Data & Embedding Engineer | Legal Document Collection, Chunking, Embeddings (`build_embeddings.py`) |
| **Anoshey Atiq** | Retrieval & Vector DB Engineer | ChromaDB Storage, `retrieval.py`, Top-k Matching |
| **Lubna Naseer** | AI Safety & Evaluation Engineer | Citation Verification, Anti-hallucination, Test Cases |

---

## 6. TECH STACK
- **Frontend / App:** Python 3.11, Streamlit 1.35+
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`), NumPy
- **LLMs:** Google Gemini 2.0 Flash, Groq LLaMA 3.3 70B
- **Web Search:** Tavily Search API
- **PDF Parsing:** pdfplumber (in-memory document upload)
- **Deployment:** Streamlit Community Cloud
- **Version Control:** GitHub

---

## 7. SUBMISSION LINKS
- 🌐 **Live App:** https://law-dost-ai.streamlit.app
- 💻 **GitHub Repo:** https://github.com/imaduddinsyed75-gif/Qanoon_Dost_AI_Assistent
