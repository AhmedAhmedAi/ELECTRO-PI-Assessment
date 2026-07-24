"""Every knob of the Section 2 RAG pipeline, in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- Models (newest available, verified against the live API on 2026-07-22) ---
EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIM = 768          # MRL truncation: 4x smaller index, near-full quality
ANSWER_MODEL = "gemini-3.5-flash-lite"
RERANK_MODEL = "gemini-3.5-flash-lite"
JUDGE_MODEL = "gemini-3.5-flash"

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"
SCRAPE_FILE = DATA_DIR / "scraped.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
CHROMA_DIR = INDEX_DIR / "chroma"
SEMANTIC_CACHE_FILE = INDEX_DIR / "semantic_cache.json"
PROMPT_CACHE_FILE = INDEX_DIR / "prompt_cache.json"

# --- Knowledge base: electropi.ai (fixed page list — the site is small,
#     a BFS crawler would be ceremony; see NOTES.md) ---
PAGES = [
    {"url": "https://electropi.ai/", "category": "company"},
    {"url": "https://electropi.ai/about", "category": "company"},
    {"url": "https://electropi.ai/services", "category": "services"},
    {"url": "https://electropi.ai/solutions", "category": "services"},
    {"url": "https://electropi.ai/case-studies", "category": "case-studies"},
    {"url": "https://electropi.ai/outsourcing", "category": "services"},
    {"url": "https://electropi.ai/ecosystem", "category": "company"},
    {"url": "https://electropi.ai/contact", "category": "company"},
    {"url": "https://electropi.ai/blog/ocr-technology-enterprise-automation", "category": "blog"},
    {"url": "https://electropi.ai/blog/ai-chatbots-for-enterprise", "category": "blog"},
    {"url": "https://electropi.ai/blog/voice-ai-solutions", "category": "blog"},
    {"url": "https://electropi.ai/blog/voice-ai-agent-for-customer-service", "category": "blog"},
]
REQUEST_DELAY = 0.5          # seconds between page fetches (politeness)
MIN_SECTION_CHARS = 60       # sections shorter than this are navigation noise

# --- Chunking ---
CHUNK_CHARS = 1200           # ~300 tokens per child chunk
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 80

# --- Retrieval ---
N_QUERY_VARIANTS = 2         # expansions on top of the original question
DENSE_TOP_K = 10             # per query variant
BM25_TOP_K = 10
RRF_K = 60                   # rank-fusion constant (standard value)
RERANK_CANDIDATES = 10       # what the reranker sees after RRF
RERANK_KEEP = 4              # what the answerer sees
# Guardrail: measured, not guessed (threshold-tuning step in analysis.ipynb, 2026-07-22:
# on-topic best-cosines 0.725-0.857, off-topic 0.533-0.651 — gate sits in
# the middle of the separation band). See NOTES.md.
DENSE_GATE = 0.68            # below this best-cosine, skip straight to refusal
RERANK_GATE = 4              # rerank relevance (0-10) a chunk must reach to count

# --- Caches ---
SEMANTIC_CACHE_THRESHOLD = 0.92
PROMPT_CACHE_TTL_SECONDS = 3600


def require_key():
    if not GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY missing — put it in section2_rag/.env")
