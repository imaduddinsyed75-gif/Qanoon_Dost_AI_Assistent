from retrieval import retrieve
question = "Under federal law, my employer delayed my salary and made deductions from my wages. What can I do?"
results = retrieve(
    question,
    top_k=5
)

print("\nQuestion:")
print(question)

print("\nRetrieved results:\n")

for i, result in enumerate(results, start=1):

    print("=" * 70)
    print("Rank:", i)
    print("Score:", result["score"])
    print("Law:", result["law_name"])
    print("Category:", result["category"])
    print("Jurisdiction:", result["jurisdiction"])
    print("Year:", result["year"])

    if result["page_start"] == result["page_end"]:
        print("Page:", result["page_start"])
    else:
        print(
            "Pages:",
            f'{result["page_start"]}-{result["page_end"]}'
        )

    print()
    print(result["text"])
    print()