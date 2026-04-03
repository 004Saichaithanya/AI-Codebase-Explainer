# AI Codebase Explainer

A RAG (Retrieval-Augmented Generation) application built with Python, LangChain, FAISS, and Streamlit that analyzes code repositories and answers natural language questions about the codebase.

## Features
- **GitHub & Zip Support**: Ingest codebases dynamically via direct repository cloning or uploaded archives.
- **Powered by Gemini**: Uses Google's generative models (`gemini-1.5-flash` and `text-embedding-004`) for lightning-fast analysis and semantic comprehension.
- **Smart Chunking**: Splits large code files while retaining context and preventing token overflow limit limits.
- **Clean Contextual Answers**: Formats responses with exact code snippets and source file paths.

## Tech Stack
- Python 3.9+
- Streamlit
- LangChain & `langchain-google-genai`
- FAISS (Facebook AI Similarity Search)
- GitPython (for repository cloning)

## Getting Started

1. Clone or download this project folder.
2. Create and activate a Virtual Environment (recommended).
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the Application
```bash
streamlit run main.py
```

## Usage
1. Provide your Google Gemini API key securely in the sidebar.
2. Choose either "GitHub Repository" or "Upload Zip File" to load the target code you want to interrogate.
3. Click "Fetch and Index" / "Extract and Index" and wait for the embeddings to generate.
4. Type natural language queries into the main window like "Where are the API routes defined?" or "Explain the authentication flow."
