import os
import streamlit as st
import tempfile
import zipfile

from ingest import load_documents_from_directory, build_vector_store
from retriever import get_qa_chain
from utils import clone_github_repo, cleanup_temp_dir
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

# --- UI Setup ---
st.set_page_config(page_title="AI Codebase Explainer", layout="wide", page_icon="🤖")
st.title("🤖 AI Codebase Explainer")
st.markdown("Ask natural language questions about your codebase, powered by Gemini and LangChain.")

# --- Session State ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'temp_repo_dir' not in st.session_state:
    st.session_state.temp_repo_dir = None

# --- Sidebar ---
st.sidebar.header("Configuration")
if not google_api_key:
    st.sidebar.error("❌ GOOGLE_API_KEY not found in .env file. Please add it to secure the app.")
else:
    st.sidebar.success("✅ Secure API Key loaded from .env")

st.sidebar.subheader("Load Codebase")
source_type = st.sidebar.radio("Source Type", ["GitHub Repository", "Upload Zip File"])

def clear_chat():
    st.session_state.chat_history = []

def process_directory(directory_path, api_key):
    with st.spinner("Loading codebase..."):
        try:
            docs = load_documents_from_directory(directory_path)
            if not docs:
                st.error("No valid code files found in the repository.")
                return False
                
            st.info(f"Loaded {len(docs)} valid files. Generating embeddings...")
            
            # Build vector store
            st.session_state.vector_store = build_vector_store(docs, api_key)
            clear_chat()
            return True
        except Exception as e:
            st.error(f"Error indexing codebase: {e}")
            return False

if source_type == "GitHub Repository":
    repo_url = st.sidebar.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")
    if st.sidebar.button("Fetch and Index", use_container_width=True):
        if not google_api_key:
            st.sidebar.error("API key is missing.")
        elif not repo_url:
            st.sidebar.error("Please enter a valid GitHub URL.")
        else:
            with st.spinner("Cloning repository..."):
                try:
                    if st.session_state.temp_repo_dir:
                        cleanup_temp_dir(st.session_state.temp_repo_dir)
                    
                    temp_dir = clone_github_repo(repo_url)
                    st.session_state.temp_repo_dir = temp_dir
                    
                    if process_directory(temp_dir, google_api_key):
                        st.sidebar.success("Repository indexed successfully!")
                except Exception as e:
                    st.sidebar.error(str(e))

elif source_type == "Upload Zip File":
    uploaded_file = st.sidebar.file_uploader("Upload Code Zip", type=["zip"])
    if st.sidebar.button("Extract and Index", use_container_width=True):
        if not google_api_key:
            st.sidebar.error("API key is missing.")
        elif not uploaded_file:
            st.sidebar.error("Please upload a file.")
        else:
            with st.spinner("Extracting..."):
                try:
                    if st.session_state.temp_repo_dir:
                        cleanup_temp_dir(st.session_state.temp_repo_dir)
                    
                    temp_dir = tempfile.mkdtemp(prefix="ai_code_explainer_zip_")
                    st.session_state.temp_repo_dir = temp_dir
                    
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        
                    if process_directory(temp_dir, google_api_key):
                        st.sidebar.success("Codebase indexed successfully!")
                except Exception as e:
                    st.sidebar.error(f"Failed to extract or index zip: {e}")

# --- Main QA Section ---

if st.session_state.vector_store is None:
    st.info("👈 Please enter your Gemini API key and load a codebase from the sidebar to start asking questions.")
else:
    # Button to clear chat
    if st.button("Clear Chat History"):
        clear_chat()
        st.rerun()
        
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            
    # Input box
    user_input = st.chat_input("Ask a question about the codebase...")
    if user_input:
        if not google_api_key:
            st.error("Please provide an API key in the sidebar.")
        else:
            # Add user msg to history and display
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        qa_chain = get_qa_chain(st.session_state.vector_store, google_api_key)
                        answer = qa_chain.invoke(user_input)
                        
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"Error generating answer: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
