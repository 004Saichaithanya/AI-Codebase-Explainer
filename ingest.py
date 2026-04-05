# ingest.py
import os
import time
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import is_valid_file
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, 
    FAISS_INDEX_PATH, EMBEDDING_BATCH_SIZE, 
    RATE_LIMIT_DELAY, MAX_RETRIES
)

def get_project_hash(identifier: str) -> str:
    """Generate a unique MD5 hash for caching based on repo URL or filename."""
    return hashlib.md5(identifier.encode('utf-8')).hexdigest()

def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Recursively load all supported files from a directory into LangChain Documents."""
    documents = []
    base_dir = Path(directory_path)
    
    for filepath in base_dir.rglob("*"):
        if filepath.is_file() and is_valid_file(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                relative_path = os.path.relpath(filepath, base_dir)
                header = f"File: {relative_path}\n"
                doc = Document(
                    page_content=header + content,
                    metadata={"source": relative_path}
                )
                documents.append(doc)
            except Exception:
                # Skip files that can't be read as utf-8
                pass
                
    return documents

def build_vector_store(documents: list[Document], google_api_key: str, identifier: str = "local_upload"):
    """Split documents, batch embed with exponential backoff, and cache to FAISS."""
    project_hash = get_project_hash(identifier)
    index_path = os.path.join(FAISS_INDEX_PATH, project_hash)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key
    )

    # 1. Check local cache first
    if os.path.exists(index_path):
        print(f"Loading cached vector store for {identifier}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if not documents:
        raise ValueError("No valid code files found in the specified directory.")
        
    # 2. Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Process in batches with retry logic
    vector_store = None
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        retries = 0
        
        while retries <= MAX_RETRIES:
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                
                # Apply standard rate limit delay if more batches remain
                if i + EMBEDDING_BATCH_SIZE < total_chunks:
                    time.sleep(RATE_LIMIT_DELAY)
                break # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resourceexhausted" in error_str or "quota" in error_str:
                    wait_time = (2 ** retries) * 5 # Exponential backoff: 5s, 10s, 20s...
                    print(f"Rate limit hit. Retrying in {wait_time}s (Attempt {retries + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    raise e # Re-raise unrelated errors
        
        if retries > MAX_RETRIES:
            raise Exception("Max retries exceeded due to rate limits. Please try again later.")

    # 4. Save to local cache
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(index_path)
    print(f"Saved new vector store to cache: {index_path}")
    
    return vector_store