from pathlib import Path
import json
import re
import pymupdf

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "extracted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in DATA_DIR.glob("*.pdf"):
    doc = pymupdf.open(pdf_path)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", pdf_path.stem).strip("_")

    jsonl_path = OUTPUT_DIR / f"{safe_stem}.jsonl"
    txt_path = OUTPUT_DIR / f"{safe_stem}.txt"

    with jsonl_path.open("w", encoding="utf-8") as jsonl, \
         txt_path.open("w", encoding="utf-8") as txt:

        for page_index, page in enumerate(doc):
            page_number = page_index + 1

            # sort=True gives a more natural top-to-bottom reading order.
            page_text = page.get_text("text", sort=True).strip()

            record = {
                "document": pdf_path.name,
                "page": page_number,
                "text": page_text
            }

            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            txt.write(
                f"===== DOCUMENT: {pdf_path.name} | PAGE: {page_number} =====\n"
            )
            txt.write(page_text)
            txt.write("\n\n")

    doc.close()

print("PDF extraction completed.")
