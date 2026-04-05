import os
import streamlit as st
import tempfile
import zipfile

from ingest import load_documents_from_directory, build_vector_store
from retriever import get_qa_chain
from utils import clone_github_repo, cleanup_temp_dir
from dotenv import load_dotenv
from st_copy_to_clipboard import st_copy_to_clipboard

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
st.sidebar.header("📂 Load Codebase")
source_type = st.sidebar.radio("Select source:", ["GitHub Repository", "Upload Zip File"], label_visibility="collapsed")

def clear_chat():
    st.session_state.chat_history = []

# main.py (Excerpt to replace)

def process_directory(directory_path, api_key, identifier):
    with st.spinner(f"Loading codebase... (This may take a moment to respect API rate limits)"):
        try:
            docs = load_documents_from_directory(directory_path)
            if not docs:
                st.error("No valid code files found in the repository.")
                return False
                
            st.info(f"Loaded {len(docs)} valid files. Generating embeddings...")
            
            # Pass the identifier to build_vector_store for caching
            st.session_state.vector_store = build_vector_store(docs, api_key, identifier)
            clear_chat()
            return True
        except Exception as e:
            st.error(f"Error indexing codebase: {e}")
            return False

if source_type == "GitHub Repository":
    repo_url = st.sidebar.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")
    if st.sidebar.button("Fetch and Index", use_container_width=True):
        # ... validation checks ...
        with st.spinner("Cloning repository..."):
            try:
                if st.session_state.temp_repo_dir:
                    cleanup_temp_dir(st.session_state.temp_repo_dir)
                
                temp_dir = clone_github_repo(repo_url)
                st.session_state.temp_repo_dir = temp_dir
                
                # Pass repo_url as the identifier
                if process_directory(temp_dir, google_api_key, repo_url):
                    st.sidebar.success("Repository indexed successfully!")
            except Exception as e:
                st.sidebar.error(str(e))

elif source_type == "Upload Zip File":
    uploaded_file = st.sidebar.file_uploader("Upload Code Zip", type=["zip"])
    if st.sidebar.button("Extract and Index", use_container_width=True):
        # ... validation checks ...
        with st.spinner("Extracting..."):
            try:
                # ... extraction logic ...
                # Pass the uploaded file name as the identifier
                if process_directory(temp_dir, google_api_key, uploaded_file.name):
                    st.sidebar.success("Codebase indexed successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to extract or index zip: {e}")
# --- Main QA Section ---

if st.session_state.vector_store is None:
    # --- New, cleaner Home Page UI ---
    st.markdown("### 👋 Welcome to your AI Code Assistant!")
    st.markdown(
        "I'm ready to help you navigate, understand, and debug your code. "
        "**Load a codebase from the sidebar to get started.**"
    )
    
    st.divider()
    
    # Feature highlights using columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔍 Deep Search")
        st.write("Ask questions about complex logic, overall architecture, or track down bugs across the entire repository.")
        
    with col2:
        st.markdown("#### 💡 Code Explainers")
        st.write("Get plain-English explanations of undocumented functions, API routes, and complex algorithms.")
        
    with col3:
        st.markdown("#### ⚡ Powered by Gemini")
        st.write("Leveraging fast, high-context embeddings and LLMs to understand your project structure instantly.")
        
    st.divider()
else:
    # Button to clear chat
    if st.button("Clear Chat History"):
        clear_chat()
        st.rerun()
        
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            # Add a copy button specifically under the assistant's responses
            if chat["role"] == "assistant":
                st_copy_to_clipboard(text=chat["content"], before_copy_label="📋 Copy Response", key=f"copy_history_{i}")
            
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
                        
                        # Add copy button for the newly generated response
                        st_copy_to_clipboard(text=answer, before_copy_label="📋 Copy Response", key=f"copy_new_{len(st.session_state.chat_history)}")
                        
                    except Exception as e:
                        error_msg = f"Error generating answer: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
