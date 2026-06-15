"""
config.py – Central configuration for the LLaMA-4 RAG Chatbot.

All settings are read from environment variables (set via .env locally
or via Streamlit Cloud secrets on deployment). API keys are intentionally
NOT read here; they are read lazily from os.environ by llm_utils.py after
Streamlit has had a chance to inject them from st.secrets.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Always load .env relative to this file's directory, regardless of CWD
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

# ─── Paths ────────────────────────────────────────────────────────────────────
DOCS_PATH: str = os.getenv("DOCS_PATH", "docs")

# ─── Qdrant Cloud ─────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

# ─── Groq / LLM ───────────────────────────────────────────────────────────────
GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─── Adaptive Chunking (characters, not tokens) ───────────────────────────────
# Different document types benefit from different chunk sizes:
#   PDFs  → longer (technical/article content)
#   DOCX  → medium (FAQs, HR, structured docs)
#   TXT   → shorter (notes, plain text)
CHUNK_SIZES: dict = {
    ".pdf":  1000,
    ".docx": 600,
    ".txt":  500,
}
CHUNK_SIZE: int = 800          # fallback for unknown types
CHUNK_OVERLAP_RATIO: float = 0.15   # 15% overlap of each chunk size
CHUNK_OVERLAP: int = 150       # fallback overlap

# ─── Retrieval settings ───────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 10      # candidates retrieved from FAISS + BM25
RERANK_TOP_N: int = 5          # best chunks kept after cross-encoder re-ranking

# ─── LLM Defaults ─────────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TEMPERATURE: float = 0.5

# ─── Recent-query detection keywords ─────────────────────────────────────────
RECENT_KEYWORDS: list[str] = [
    "2024", "2025", "2026",
    "latest", "current", "now",
    "this year", "today", "recently",
    "new", "breaking", "live",
]

# ─── Supported document extensions ───────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".docx")
