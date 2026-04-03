import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import is_valid_file
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Recursively load all supported files from a directory into LangChain Documents."""
    documents = []
    base_dir = Path(directory_path)
    
    for filepath in base_dir.rglob("*"):
        if filepath.is_file() and is_valid_file(filepath):
            try:
                # Read file content safely
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Make relative path for prettier display
                relative_path = os.path.relpath(filepath, base_dir)
                
                header = f"File: {relative_path}\n"
                doc = Document(
                    page_content=header + content,
                    metadata={"source": relative_path}
                )
                documents.append(doc)
            except Exception as e:
                # Skip files that can't be read as utf-8
                pass
                
    return documents

def build_vector_store(documents: list[Document], google_api_key: str):
    """Split documents into chunks and create a FAISS vector store."""
    if not documents:
        raise ValueError("No valid code files found in the specified directory.")
        
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize Google Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key
    )
    
    # Create FAISS vectorstore
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store
