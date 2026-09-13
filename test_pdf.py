import pdfplumber

pdf_path = "data/raw/federal/labor/payment_of_wages_act_1936.pdf"
output_path = "data/processed/payment_of_wages_act_1936.txt"

full_text = ""

with pdfplumber.open(pdf_path) as pdf:
    print("Total pages:", len(pdf.pages))

    for page_number, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()

        if text:
            full_text += f"\n\n--- PAGE {page_number} ---\n\n"
            full_text += text

with open(output_path, "w", encoding="utf-8") as file:
    file.write(full_text)

print("Extraction complete.")
print("Saved to:", output_path)
print("Total characters:", len(full_text))