import pymupdf
import json

pdf_path = "annual_report.pdf"

# Open the PDF
doc = pymupdf.open(pdf_path)

# Chunk settings
CHUNK_SIZE = 1000
OVERLAP = 200

chunks = []

for page_number, page in enumerate(doc, start=1):
    text = page.get_text().strip()

    if not text:
        continue

    # Split text into smaller chunks
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

        start += CHUNK_SIZE - OVERLAP

doc.close()

# Save chunks and their page numbers
with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Chunking completed!")
print("Total chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0]["text"] if chunks else "No text extracted")
print("\nSource page:", chunks[0]["page"] if chunks else "N/A")