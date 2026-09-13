"""
Phase 3 + Phase 8 — LLM Integration & Web Search Fallback for Qanoon Dost AI

Connects Phase 2's retrieval.py (legal knowledge base) to Gemini (primary)
and Groq (fallback) to generate simplified, source-grounded legal answers.

Phase 8 addition: optional Tavily web search fallback when local retrieval
confidence is low and the user has enabled the Live Web Search toggle.

Usage:
    from llm_response import get_legal_answer

    result = get_legal_answer(
        "My landlord is trying to evict me. What are my rights?",
        language="roman_urdu",
        enable_web_search=True,    # Phase 8 toggle
    )

    print(result["answer"])
    print(result["sources"])
    print(result["provider_used"])
    print(result.get("web_search_used"))   # True if Tavily was used
"""

import os
from retrieval import retrieve

# --------------------------------------------------
# Configuration
# --------------------------------------------------

GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "openai/gpt-oss-120b"

DEFAULT_TOP_K = 5

# Phase 8: if the best local chunk similarity is below this threshold
# AND the user has enabled Live Web Search, fall back to Tavily.
WEB_SEARCH_CONFIDENCE_THRESHOLD = 0.30

# Phase 8: number of Tavily results to fetch
TAVILY_TOP_N = 3


# --------------------------------------------------
# Secrets helper
# --------------------------------------------------

def get_api_key(name: str) -> str | None:
    """
    Look up an API key from Streamlit Secrets first, then fall back to a
    plain environment variable.  Works the same way whether running locally,
    in Colab, or deployed on Streamlit Community Cloud.

    Priority:
      1. st.secrets  (Streamlit Cloud dashboard / .streamlit/secrets.toml)
      2. os.environ  (.env loaded by python-dotenv in local / Colab use)
    """
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # Not running inside Streamlit, or secrets.toml is absent — fall
        # back to environment variables below.
        pass

    return os.environ.get(name)


# Keys are resolved at import time so they are available to every call.
GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")
GROQ_API_KEY   = get_api_key("GROQ_API_KEY")
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY")   # Phase 8


# --------------------------------------------------
# Language instructions
# --------------------------------------------------

LANGUAGE_INSTRUCTIONS = {
    "english":     "Answer in clear, simple English.",
    "urdu":        "Answer in simple, easy-to-read Urdu script (اردو).",
    "roman_urdu":  (
        "Answer in Roman Urdu (Urdu written in English letters), "
        "in a simple, conversational tone."
    ),
}


# --------------------------------------------------
# Build context from retrieved chunks (local KB)
# --------------------------------------------------

def build_context(results: list[dict]) -> str:
    """Turn retrieval.py results into a labeled context block."""
    parts = []
    for i, r in enumerate(results, start=1):
        if r["page_start"] == r["page_end"]:
            page_info = f'Page {r["page_start"]}'
        else:
            page_info = f'Pages {r["page_start"]}-{r["page_end"]}'

        source = (
            f'{r["law_name"]} ({r["jurisdiction"]}, {r["year"]}), {page_info}'
        )
        parts.append(f"[Source {i}: {source}]\n{r['text']}")

    return "\n\n".join(parts)


# --------------------------------------------------
# Build context from Tavily web results (Phase 8)
# --------------------------------------------------

def build_web_context(web_results: list[dict]) -> str:
    """
    Turn Tavily search results into a labeled context block.

    Each result is expected to have at least:
      - title  (str)
      - url    (str)
      - content (str)
    """
    parts = []
    for i, r in enumerate(web_results, start=1):
        parts.append(
            f"[Web Source {i}: {r.get('title', 'Untitled')} — {r.get('url', '')}]\n"
            f"{r.get('content', '')}"
        )
    return "\n\n".join(parts)


# --------------------------------------------------
# Build the final prompt (local KB)
# --------------------------------------------------

def build_prompt(question: str, context: str, language: str) -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(
        language, LANGUAGE_INSTRUCTIONS["roman_urdu"]
    )

    return f"""You are Qanoon Dost AI, a legal-rights assistant for citizens of Pakistan.

Answer the user's question using ONLY the legal context provided below.
Do not use outside knowledge. Do not invent laws, sections, or facts.

Rules:
1. {lang_instruction}
2. Keep the answer simple and easy to understand for a non-lawyer.
3. If the context does not contain enough information to answer confidently,
   say so clearly instead of guessing.
4. At the end, list which sources (by law name) you used.
5. Add one short line reminding the user this is general legal information,
   not a substitute for a licensed lawyer.

Legal Context:
{context}

User's Question:
{question}

Answer:"""


# --------------------------------------------------
# Build prompt for web-search results (Phase 8)
# --------------------------------------------------

def build_web_prompt(question: str, web_context: str, language: str) -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(
        language, LANGUAGE_INSTRUCTIONS["roman_urdu"]
    )

    return f"""You are Qanoon Dost AI, a legal-rights assistant for citizens of Pakistan.

The local legal knowledge base did not contain a confident answer, so the
following information has been retrieved live from the web.

Rules:
1. {lang_instruction}
2. Keep the answer simple and easy to understand for a non-lawyer.
3. If the web results do not contain enough information, say so clearly.
4. At the end, list the web sources you used (title and URL).
5. Add this EXACT disclaimer at the very end:
   "⚠️ This information is retrieved from the web, please verify."
6. Add one short line reminding the user this is general legal information,
   not a substitute for a licensed lawyer.

Web Search Results:
{web_context}

User's Question:
{question}

Answer:"""


# --------------------------------------------------
# Gemini
# --------------------------------------------------

def call_gemini(prompt: str) -> str:
    from google import genai

    key = get_api_key("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=key.strip())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


# --------------------------------------------------
# Groq (fallback)
# --------------------------------------------------

def call_groq(prompt: str) -> str:
    from groq import Groq
    key = get_api_key("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    client = Groq(api_key=key.strip())
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def call_llm(prompt: str) -> tuple[str, str]:
    """
    Try Gemini first; fall back to Groq.

    Returns:
        (answer_text, provider_name)
    """
    try:
        return call_gemini(prompt), "gemini"
    except Exception as gemini_err:
        try:
            return call_groq(prompt), "groq"
        except Exception as groq_err:
            raise RuntimeError(
                f"Both LLM providers failed.\n"
                f"Gemini: {gemini_err}\n"
                f"Groq:   {groq_err}"
            )


# --------------------------------------------------
# Phase 8 — Tavily web search
# --------------------------------------------------

def tavily_search(query: str, top_n: int = TAVILY_TOP_N) -> list[dict]:
    """
    Search the web with Tavily and return the top `top_n` results.

    Returns a list of dicts with keys: title, url, content.
    Returns an empty list if the key is missing or the call fails.
    """
    if not TAVILY_API_KEY:
        return []

    try:
        from tavily import TavilyClient  # pip install tavily-python
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=top_n,
        )
        return response.get("results", [])
    except Exception:
        return []


# --------------------------------------------------
# Main function
# --------------------------------------------------

def get_legal_answer(
    question: str,
    language: str = "roman_urdu",
    top_k: int = DEFAULT_TOP_K,
    enable_web_search: bool = False,
) -> dict:
    """
    Full pipeline (Phases 3 + 8):

    1. Retrieve relevant legal chunks from the local KB (Phase 2).
    2. If local confidence is high enough (or web search is disabled),
       build a grounded prompt and call Gemini / Groq.
    3. [Phase 8] If local confidence is LOW and `enable_web_search` is True,
       fall back to Tavily and answer from web results instead.

    Returns:
        {
            "answer":         str,
            "sources":        [ {law_name, jurisdiction, year, ...}, ... ],
            "provider_used":  "gemini" | "groq" | None,
            "question":       str,
            "web_search_used": bool,   # Phase 8
        }
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    question = question.strip()

    # ── Local retrieval ──────────────────────────────────────────────────
    results = retrieve(question, top_k=top_k)

    best_score = max((r["score"] for r in results), default=0.0)

    # ── Phase 8: decide whether to use web search ────────────────────────
    use_web = (
        enable_web_search
        and best_score < WEB_SEARCH_CONFIDENCE_THRESHOLD
    )

    # ── Web-search branch ────────────────────────────────────────────────
    if use_web:
        web_results = tavily_search(question)

        if web_results:
            web_context = build_web_context(web_results)
            web_prompt  = build_web_prompt(question, web_context, language)
            answer, provider = call_llm(web_prompt)

            # Ensure the disclaimer is always present even if the LLM
            # omits it (defensive).
            disclaimer = (
                "\n\n⚠️ **This information is retrieved from the web, please verify.**"
            )
            if "please verify" not in answer.lower():
                answer += disclaimer

            return {
                "answer":          answer,
                "sources":         [],          # no local law sources
                "provider_used":   provider,
                "question":        question,
                "web_search_used": True,
            }

    # ── Local KB branch ──────────────────────────────────────────────────
    if not results:
        return {
            "answer":          (
                "Maaf kijiye, is sawal ke liye koi relevant legal "
                "information knowledge base mein nahi mili."
            ),
            "sources":         [],
            "provider_used":   None,
            "question":        question,
            "web_search_used": False,
        }

    context = build_context(results)
    prompt  = build_prompt(question, context, language)
    answer, provider = call_llm(prompt)

    sources = [
        {
            "law_name":    r["law_name"],
            "jurisdiction": r["jurisdiction"],
            "year":        r["year"],
            "page_start":  r["page_start"],
            "page_end":    r["page_end"],
            "score":       r["score"],
        }
        for r in results
    ]

    return {
        "answer":          answer,
        "sources":         sources,
        "provider_used":   provider,
        "question":        question,
        "web_search_used": False,
    }
