# ⚖️ Qanoon Dost AI (RightsGuide AI)

> An AI-powered legal assistant providing precise, citation-grounded guidance on Pakistan's legal framework using Retrieval-Augmented Generation (RAG).

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://law-dost-ai.streamlit.app)

---

## 📌 Overview
**Qanoon Dost AI** bridges the access-to-justice gap by translating complex Pakistani statutory frameworks into actionable, plain-language guidance in **English, Urdu, and Roman Urdu**. 

Built on an end-to-end RAG architecture, it minimizes hallucinations by anchoring responses directly to official legal statutes and providing verifiable section-level citations.

---

## 🎯 Legal Coverage
- **Tenant & Rent Laws**: Punjab Rented Premises Act 2009, Sindh Rented Premises Ordinance 1979, West Pakistan Urban Rent Restriction Ordinance 1959.
- **Family & Inheritance**: Muslim Family Laws Ordinance 1961, custody and succession guidelines.
- **Criminal Procedure**: Code of Criminal Procedure (CrPC) 1898, FIR filing procedures.
- **Consumer Protection**: Punjab Consumer Protection Act 2005.
- **Labor & Employment**: Payment of Wages Act 1936, Minimum Wages Act, Shops & Establishments Ordinance.

---

## 🛠️ System Architecture
1. **Document Ingestion & Chunking**: Legal statutes processed into structured text chunks with section and page metadata.
2. **Dense Vector Retrieval**: Chunk representations generated via SentenceTransformers to enable semantic similarity matching.
3. **LLM Generation Engine**: High-throughput inference orchestrated via Groq API (`openai/gpt-oss-120b`) with prompt constraints enforcing statutory citations.
4. **Interactive UI**: Multi-turn conversational interface built on Streamlit with persistent session storage and multilingual toggles.

---

## 🚀 Live Demo
Access the live application 24/7 at: **[law-dost-ai.streamlit.app](https://law-dost-ai.streamlit.app)**

---

## ⚙️ Local Setup

```bash
# Clone the repository
git clone [https://github.com/imaduddinsyed75-gif/Qanoon_Dost_AI_Assistent.git](https://github.com/imaduddinsyed75-gif/Qanoon_Dost_AI_Assistent.git)
cd Qanoon_Dost_AI_Assistent

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
