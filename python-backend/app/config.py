"""Settings loaded from environment variables (.env file)."""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Same embedding model as the Node.js version (Xenova/all-MiniLM-L6-v2 is a port of
# this model), so the existing Pinecone index keeps working: 384-dim, normalized.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))  # extra search attempts when chunks look irrelevant
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "500"))

NOT_FOUND_ANSWER = (
    "I don't have information about this in my knowledge base. "
    "Please visit the official government website."
)
