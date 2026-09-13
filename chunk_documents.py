import json
import re

METADATA_FILE = "metadata.json"
OUTPUT_FILE = "chunks.json"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250


def split_by_pages(text):
    pattern = r"--- PAGE (\d+) ---"
    parts = re.split(pattern, text)

    pages = []

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = parts[i + 1].strip()

        pages.append({
            "page": page_number,
            "text": page_text
        })

    return pages


def looks_like_contents_page(text):
    """
    Detect table-of-contents style pages even when the literal
    word CONTENTS is missing.
    """

    upper = text.upper()

    if "TABLE OF CONTENTS" in upper or "CONTENTS" in upper:
        return True

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    numbered_heading_count = 0
    short_line_count = 0

    for line in lines:
        if re.match(r"^\d+\.\s+[A-Z]", line):
            numbered_heading_count += 1

        if len(line) < 80:
            short_line_count += 1

    # Typical contents page:
    # many numbered short headings and very little prose
    if (
        numbered_heading_count >= 8
        and short_line_count / len(lines) > 0.75
    ):
        return True

    return False


def clean_line(line):
    line = line.strip()

    # Normalize whitespace
    line = re.sub(r"[ \t]+", " ", line)

    # Fix footnote artifacts such as:
    # 1This Act was...
    line = re.sub(
        r"^(\d+)([A-Z])",
        r"\1 \2",
        line
    )

    return line


def is_structural_line(line):
    """
    Detect legal headings / section starts that should begin
    a new logical paragraph.
    """

    patterns = [
        r"^CHAPTER\b",
        r"^PART\b",
        r"^SCHEDULE\b",
        r"^THE\s+[A-Z]",
        r"^\d+\.\s+[A-Z]",
        r"^\([a-zA-Z0-9]+\)\s+",
    ]

    return any(
        re.match(pattern, line)
        for pattern in patterns
    )


def rebuild_paragraphs(page_number, text):
    """
    Reconstruct PDF-wrapped lines into readable legal paragraphs.

    Every paragraph keeps the page where it started and ended.
    """

    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    paragraphs = []

    current_text = ""
    current_start_page = page_number
    current_end_page = page_number

    for line in lines:

        if is_structural_line(line):

            if current_text:
                paragraphs.append({
                    "text": current_text.strip(),
                    "page_start": current_start_page,
                    "page_end": current_end_page
                })

            current_text = line
            current_start_page = page_number
            current_end_page = page_number

        else:
            if current_text:
                current_text += " " + line
            else:
                current_text = line
                current_start_page = page_number

            current_end_page = page_number

    if current_text:
        paragraphs.append({
            "text": current_text.strip(),
            "page_start": current_start_page,
            "page_end": current_end_page
        })

    return paragraphs


def merge_cross_page_continuations(paragraphs):
    """
    Merge a paragraph that obviously continues onto the next page.

    Example:
    page 6 ends:
        "...for the time"

    page 7 starts:
        "being in force..."

    These should become one paragraph.
    """

    if not paragraphs:
        return []

    merged = [paragraphs[0]]

    for paragraph in paragraphs[1:]:

        previous = merged[-1]

        previous_text = previous["text"]
        current_text = paragraph["text"]

        previous_ends_sentence = bool(
            re.search(r'[.;:?!)”\]]$', previous_text)
        )

        current_starts_structure = is_structural_line(
            current_text
        )

        # If previous text looks unfinished and next text is not
        # a new legal section, join them.
        if (
            not previous_ends_sentence
            and not current_starts_structure
        ):
            previous["text"] = (
                previous_text + " " + current_text
            )

            previous["page_end"] = paragraph["page_end"]

        else:
            merged.append(paragraph)

    return merged


def create_chunk(paragraphs):
    text = "\n".join(
        paragraph["text"]
        for paragraph in paragraphs
    ).strip()

    return {
        "text": text,
        "page_start": paragraphs[0]["page_start"],
        "page_end": paragraphs[-1]["page_end"]
    }


def get_overlap_paragraphs(paragraphs):
    """
    Preserve overlap using COMPLETE paragraphs.
    """

    overlap = []
    total_length = 0

    for paragraph in reversed(paragraphs):

        overlap.insert(0, paragraph)

        total_length += len(paragraph["text"]) + 1

        if total_length >= CHUNK_OVERLAP:
            break

    return overlap


def split_into_chunks(paragraphs):
    """
    Build chunks from complete legal paragraphs.
    """

    chunks = []

    current = []
    current_length = 0

    for paragraph in paragraphs:

        paragraph_length = len(paragraph["text"]) + 1

        if (
            current
            and current_length + paragraph_length > CHUNK_SIZE
        ):
            chunks.append(
                create_chunk(current)
            )

            current = get_overlap_paragraphs(current)

            current_length = sum(
                len(item["text"]) + 1
                for item in current
            )

        current.append(paragraph)

        current_length += paragraph_length

    if current:
        chunks.append(
            create_chunk(current)
        )

    return chunks


# ==================================================
# Load metadata
# ==================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)


all_chunks = []
chunk_id = 1


# ==================================================
# Process documents
# ==================================================

for document in documents:

    print("Processing:", document["law_name"])

    file_path = document["processed_file"]

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    pages = split_by_pages(text)

    document_paragraphs = []

    for page in pages:

        if looks_like_contents_page(page["text"]):
            continue

        page_paragraphs = rebuild_paragraphs(
            page["page"],
            page["text"]
        )

        document_paragraphs.extend(
            page_paragraphs
        )

    # Repair continuations across page boundaries
    document_paragraphs = merge_cross_page_continuations(
        document_paragraphs
    )

    document_chunks = split_into_chunks(
        document_paragraphs
    )

    for chunk in document_chunks:

        all_chunks.append({
            "chunk_id": chunk_id,
            "law_name": document["law_name"],
            "category": document["category"],
            "jurisdiction": document["jurisdiction"],
            "year": document["year"],

            # Keep compatibility with older code
            "page": chunk["page_start"],

            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],

            "text": chunk["text"]
        })

        chunk_id += 1


# ==================================================
# Save
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


print()
print("Chunking complete.")
print("Total chunks:", len(all_chunks))
print("Saved to:", OUTPUT_FILE)