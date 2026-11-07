"""
Central configuration for THB StudentAssist.

All values can be overridden through environment variables or a `.env` file
in the project root (see `.env.example`).
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_DIR = os.getenv("CLEAN_DIR", "clean_documents")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vectorstore")
PDF_DIR = os.getenv("PDF_DIR", "sources/pdf")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thb_studentassist")

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# "ollama": talk to Ollama directly. "openai": any OpenAI-compatible endpoint
# (LiteLLM gateway, vLLM, ...). See llm_backend.py and docker-compose.yml.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:18080")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:4000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1-8b")  # model name as exposed by the gateway
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# Ollama defaults to a 2048-token context window and silently truncates the
# prompt beyond that. Our prompt (instructions + 6 chunks + history) is larger.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "8"))              # chunks passed to the LLM
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "25"))  # candidates per retriever before fusion
# The default embedding model only reads the first 128 tokens (~450 German
# characters) of a chunk, so chunks must stay small enough to be fully embedded.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ── Conversation ──────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 60 * 60)))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))

# ── Web app ───────────────────────────────────────────────────────────────────
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
COOKIE_SECURE = _bool("COOKIE_SECURE", False)  # set True when served over HTTPS

DEBUG_RETRIEVAL = _bool("DEBUG_RETRIEVAL", False)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
