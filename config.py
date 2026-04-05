# config.py
import os

# Text chunking configuration - Increased to reduce API calls
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Rate limiting and Batching config
EMBEDDING_BATCH_SIZE = 15
RATE_LIMIT_DELAY = 10 # seconds to wait between batches
MAX_RETRIES = 5 # max retries for 429 exponential backoff

# Supported file extensions to parse
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", 
    ".rs", ".ts", ".tsx", ".jsx", ".html", ".css", ".md", ".json"
}

# Directories and files to ignore during document loading
IGNORED_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".vscode", 
    ".idea", "build", "dist", ".next", ".venv", "env", "ENV", 
    ".DS_Store", "coverage"
}
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", 
    "requirements.txt", ".env" 
}

# FAISS index storage path (caching)
FAISS_INDEX_PATH = "faiss_index"

# Embedding model config
EMBEDDING_MODEL = "models/gemini-embedding-001" # Updated based on README

# LLM model config
LLM_MODEL = "gemini-2.5-flash" # Updated based on README