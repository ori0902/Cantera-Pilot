"""Central configuration. Edit MODEL to match what you pulled via Ollama."""
from pathlib import Path

# ----- LLM -----
MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"

MAX_TOKENS = {"planner": 512, "codegen": 1024, "analysis": 512}
TEMPERATURE = {"planner": 0.0, "codegen": 0.1, "analysis": 0.3}

# ----- Agent loop -----
MAX_CODEGEN_RETRIES = 3
EXECUTION_TIMEOUT_SEC = 180      # flame_speed solves can take 30-60s, so be generous

# ----- Paths -----
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
REFERENCE_DIR = ROOT / "reference"
LOG_DIR = ROOT / "logs"
CHROMA_DIR = ROOT / "rag" / "chroma_db"

# ----- RAG -----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_LINES = 5
RETRIEVAL_K = 8
RERANK_K = 3

# ----- Execution sandbox -----
ALLOWED_IMPORTS = {
    "cantera", "numpy", "np", "math", "json", "sys",
}
