from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

def get_qa_chain(vector_store, google_api_key: str):
    """Create a retrieval QA chain using LCEL and Gemini LLM."""
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=google_api_key,
        temperature=0.2
    )

    # Setup the prompt template
    system_prompt = (
        "You are an expert AI software engineer tasked with explaining a codebase. "
        "Use the following pieces of retrieved context containing code snippets and file paths to answer the user's question.\n"
        "If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.\n"
        "Always mention the source file name(s) and include relevant code snippets in your answer to provide a helpful and educational response.\n"
        "\n"
        "Context:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Connect the vectorstore as a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Combine into a final retrieval chain
    qa_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain
