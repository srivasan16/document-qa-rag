from pathlib import Path
import json, re

DATA_DIR = Path("data")
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WORDS = 800
OVERLAP_WORDS = 100
MAX_PARAGRAPH_WORDS = 1000

def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()

def words(text):
    return text.split()

def fixed_chunks(page_records):
    result, chunk_id = [], 0
    for rec in page_records:
        ws = words(clean_text(rec["text"]))
        start = 0
        while start < len(ws):
            end = min(start + TARGET_WORDS, len(ws))
            chunk_id += 1
            result.append({
                "chunk_id": f"fixed_{chunk_id:06d}",
                "strategy": "fixed_size",
                "document": rec["document"],
                "page": rec["page"],
                "text": " ".join(ws[start:end]),
                "word_count": end - start
            })
            if end >= len(ws):
                break
            start = max(end - OVERLAP_WORDS, start + 1)
    return result

def split_paragraphs(text):
    text = clean_text(text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    if len(blocks) > 1:
        return blocks
    return [x.strip() for x in text.splitlines() if x.strip()]

def paragraph_chunks(page_records):
    result, chunk_id = [], 0
    for rec in page_records:
        paras = split_paragraphs(rec["text"])
        current = []

        def flush():
            nonlocal chunk_id, current
            if current:
                chunk_id += 1
                txt = "\n\n".join(current)
                result.append({
                    "chunk_id": f"paragraph_{chunk_id:06d}",
                    "strategy": "paragraph_based",
                    "document": rec["document"],
                    "page": rec["page"],
                    "text": txt,
                    "word_count": len(txt.split())
                })
                current = []

        for para in paras:
            pw = len(para.split())
            if current and len("\n\n".join(current).split()) + pw > TARGET_WORDS:
                flush()
            if pw <= MAX_PARAGRAPH_WORDS:
                current.append(para)
            else:
                # Safety fallback for oversized PDF blocks.
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    if current and len("\n\n".join(current).split()) + len(sentence.split()) > TARGET_WORDS:
                        flush()
                    current.append(sentence)
        flush()
    return result

def write_jsonl(items, path):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

pages = []
for path in sorted(EXTRACTED_DIR.glob("*.jsonl")):
    with path.open(encoding="utf-8") as f:
        pages.extend(json.loads(line) for line in f if line.strip())

fixed = fixed_chunks(pages)
paragraph = paragraph_chunks(pages)

write_jsonl(fixed, CHUNKS_DIR / "fixed_size_chunks.jsonl")
write_jsonl(paragraph, CHUNKS_DIR / "paragraph_chunks.jsonl")

print(f"Pages: {len(pages)}")
print(f"Fixed-size chunks: {len(fixed)}")
print(f"Paragraph chunks: {len(paragraph)}")
print("Chunking completed.")
