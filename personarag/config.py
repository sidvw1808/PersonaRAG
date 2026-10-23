"""Central configuration. Every value can be overridden from the environment or a .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PERSONAS_DIR = ROOT / "personas"
DB_DIR = ROOT / "data" / "chroma"

# Embeddings: bge-small is small enough for CPU and handles 512-token passages.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models retrieve better when queries (not passages) carry this instruction.
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

# Chunking (words, not tokens: ~300 words stays under bge's 512-token window).
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "300"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "6"))
# Hits below this cosine similarity are dropped as irrelevant.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.50"))

# LLM: "claude" (Anthropic API) or "ollama" (local model).
# Unset -> Claude when an API key is configured, otherwise the free local model.
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("claude" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")
# Server-side refusal fallback (re-routes a refused request to another model).
CLAUDE_FALLBACKS = os.getenv("CLAUDE_FALLBACKS", "1") == "1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "0") == "1"

# Speech-to-text for podcasts and caption-less videos (faster-whisper).
# "base.en" is fast on CPU; "small.en" is more accurate. Set WHISPER_DEVICE=cuda to use the GPU.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
# Transcribe a YouTube video with Whisper when its captions can't be fetched.
WHISPER_FALLBACK = os.getenv("WHISPER_FALLBACK", "1") == "1"

# Number of previous chat turns sent back to the model.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
