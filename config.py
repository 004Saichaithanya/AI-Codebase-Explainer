# config.py
import os

# Text chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Supported file extensions to parse
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", 
    ".rs", ".ts", ".tsx", ".jsx", ".html", ".css", ".md", ".json"
}

# Directories and files to ignore during document loading
IGNORED_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".vscode", 
    ".idea", "build", "dist", ".next"
}
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", 
    "requirements.txt" # optional to ignore
}

# FAISS index storage path (optional caching)
FAISS_INDEX_PATH = "faiss_index"

# Embedding model config
# We'll use "models/gemini-embedding-001" as it is the current supported model
EMBEDDING_MODEL = "models/gemini-embedding-001"

# LLM model config
# Using 'gemini-2.5-flash' as it is the current supported model
LLM_MODEL = "gemini-2.5-flash"
