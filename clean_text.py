import re
from pathlib import Path

input_path = Path(
    "data/processed/payment_of_wages_act_1936.txt"
)

output_path = Path(
    "data/processed/payment_of_wages_act_1936_clean.txt"
)

text = input_path.read_text(encoding="utf-8")

# Normalize spaces and tabs
text = re.sub(r"[ \t]+", " ", text)

# Remove excessive blank lines
text = re.sub(r"\n{3,}", "\n\n", text)

# Fix extraction artifact such as:
# 1THE WEST PAKISTAN...
text = re.sub(
    r"(?m)^\d+(THE\s+[A-Z][A-Z\s,0-9\-]+)$",
    r"\1",
    text
)

# Remove standalone page labels such as:
# Page 3 of 20
text = re.sub(
    r"(?im)^Page\s+\d+\s+of\s+\d+\s*$",
    "",
    text
)

# Remove standalone labels such as:
# 1 | Pag e
# 2 | Page
text = re.sub(
    r"(?im)^\d+\s*\|\s*Pag\s*e\s*$",
    "",
    text
)

# Clean excessive blank lines again
text = re.sub(r"\n{3,}", "\n\n", text)

output_path.write_text(
    text.strip() + "\n",
    encoding="utf-8"
)

print("Cleaning complete.")
print("Saved to:", output_path)
print("Cleaned characters:", len(text))