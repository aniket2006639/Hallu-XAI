"""Central settings. Override any of these with environment variables or a .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
MODEL_DIR = BASE_DIR / "models" / "saved"
CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.json"
CLASSIFIER_META_PATH = MODEL_DIR / "xgb_meta.json"

# Models (downloaded automatically from HuggingFace the first time)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "3"))
USE_WIKIPEDIA = os.getenv("USE_WIKIPEDIA", "true").lower() == "true"
WIKIPEDIA_TIMEOUT = float(os.getenv("WIKIPEDIA_TIMEOUT", "5"))

# Pipeline behaviour
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "12"))
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "8000"))
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"

# Verdict thresholds on the hallucination probability
HIGH_RISK = float(os.getenv("HIGH_RISK", "0.60"))   # >= this  -> likely hallucinated
LOW_RISK = float(os.getenv("LOW_RISK", "0.35"))     # <  this  -> supported
