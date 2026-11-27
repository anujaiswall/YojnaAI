import time

from app.cache import TTLCache
from ingest import chunk_text, read_pdf


def test_chunks_respect_word_limit_and_skip_tiny_fragments():
    paragraph = "word " * 250
    text = "\n\n".join([paragraph, "page 3", paragraph, paragraph])
    chunks = chunk_text(text, max_words=600)

    assert len(chunks) == 2
    assert all(len(c.split()) < 600 for c in chunks)
    assert "page 3" not in " ".join(chunks)


def test_reads_text_from_a_pdf(tmp_path):
    import pymupdf

    path = tmp_path / "scheme.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Pradhan Mantri Awas Yojana provides housing for all.")
    doc.save(path)
    doc.close()

    assert "Awas Yojana" in read_pdf(path)


def test_cache_expires_and_evicts_oldest():
    cache = TTLCache(max_items=2, ttl_seconds=60)
    cache.set("a?", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None  # evicted: least recently used
    assert cache.get("c") == 3

    short = TTLCache(ttl_seconds=0)
    short.set("q", 1)
    time.sleep(0.01)
    assert short.get("q") is None
