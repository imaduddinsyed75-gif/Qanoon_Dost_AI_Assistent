"""
Phase 3 — Test Script

Run this to test the full retrieval + LLM pipeline end-to-end.
Make sure GEMINI_API_KEY and/or GROQ_API_KEY are set in your environment
(or Colab Secrets) before running.
"""

from llm_response import get_legal_answer

TEST_QUESTIONS = [
    ("My landlord is trying to evict me. What are my rights?", "roman_urdu"),
    ("I bought a defective product and the seller refuses to refund me. What can I do?", "english"),
    ("My employer is paying me less than minimum wage.", "roman_urdu"),
]

if __name__ == "__main__":
    for question, lang in TEST_QUESTIONS:
        print("=" * 60)
        print(f"QUESTION ({lang}): {question}")
        print("=" * 60)

        result = get_legal_answer(question, language=lang)

        print(f"\nProvider used: {result['provider_used']}")
        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['law_name']} ({s['jurisdiction']}, {s['year']}), "
                  f"Page {s['page_start']}-{s['page_end']}, score={s['score']}")
        print("\n")
