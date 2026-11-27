# Yojna-AI – Python back end (FastAPI + LangGraph)

A Python rewrite of the Node.js back end in the repo root. It answers questions about
Indian government schemes from official PDFs, using the same Pinecone index and the
same embedding model, and adds three things the Node.js version doesn't have:

- **Query rewriting:** Gemini turns vague questions into clear search queries.
- **A Self-RAG relevance check (LangGraph):** before answering, Gemini checks that the
  retrieved chunks actually answer the question. If not, it rewrites the query and
  searches again; if that fails too, it says it doesn't know instead of guessing.
- **Caching and input validation:** repeat questions are served from memory, and
  requests are validated with Pydantic.

## How a question flows

```text
question
   │
   ▼
rewrite ──► retrieve (Pinecone, top 5) ──► grade: are these chunks relevant?
   ▲                                          │            │
   │        not relevant, retry left          │            │ relevant
   └──────────────────────────────────────────┘            ▼
                                                    answer from chunks only
          not relevant, no retries left ──► "I don't have information about this..."
```

The response includes the sources, the search query used, and each step the graph took.

## Project layout

```text
python-backend/
  app/
    main.py        FastAPI app: POST /api/chat, GET /health
    graph.py       LangGraph flow: rewrite → retrieve → grade → answer / retry
    llm.py         Gemini calls: rewrite, grade (structured JSON output), answer
    retrieval.py   Pinecone vector search
    embeddings.py  sentence-transformers all-MiniLM-L6-v2 (384-dim)
    cache.py       in-memory cache with expiry
    config.py      settings from .env
  ingest.py        PDFs → chunks → embeddings → Pinecone
  tests/           pytest suite (Gemini and Pinecone are faked; no keys needed)
```

## Run it

```bash
cd python-backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your Gemini and Pinecone keys

# Only needed if the index is empty. Uses the same ids as scripts/ingest.js.
python ingest.py

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs to try the API, or:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Can a small farmer get PM-KISAN money?"}'
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover the API, validation, caching, the retry path in the graph, PDF reading
and chunking. They also run on every push through GitHub Actions
(`.github/workflows/python-ci.yml`).
