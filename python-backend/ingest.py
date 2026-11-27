"""Read scheme PDFs, split them into chunks, embed them and upload to Pinecone.

Python version of scripts/ingest.js. Uses the same chunk ids and metadata
(text, source, chunkIndex), so it updates the same Pinecone index.

Usage:  python ingest.py            (reads ../data/schemes/*.pdf)
        python ingest.py --dir path/to/pdfs
"""
import argparse
import re
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "schemes"
MAX_WORDS = 600       # same limit as the Node.js chunker
MIN_PARAGRAPH = 50    # skip tiny fragments like page numbers
BATCH = 10


def read_pdf(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        return "\n\n".join(page.get_text() for page in doc)


def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Group paragraphs into chunks of up to max_words words, never splitting a paragraph."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if len(p) > MIN_PARAGRAPH]

    chunks, current = [], ""
    for paragraph in paragraphs:
        candidate = f"{current} {paragraph}".strip()
        if len(candidate.split()) < max_words:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def ingest(pdf_dir: Path) -> None:
    from app.embeddings import embed
    from app.retrieval import _index

    files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(files)} PDF(s) in {pdf_dir}")
    index = _index()

    for pdf in files:
        chunks = chunk_text(read_pdf(pdf))
        print(f"\n{pdf.name}: {len(chunks)} chunks")
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start:start + BATCH]
            vectors = [
                {
                    "id": f"{pdf.name}-chunk-{start + i}",
                    "values": values,
                    "metadata": {"text": text, "source": pdf.name, "chunkIndex": start + i},
                }
                for i, (text, values) in enumerate(zip(batch, embed(batch)))
            ]
            index.upsert(vectors=vectors, show_progress=False)
            print(f"  stored chunks {start}-{start + len(batch) - 1}")
    print("\nAll PDFs ingested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ingest(parser.parse_args().dir)
