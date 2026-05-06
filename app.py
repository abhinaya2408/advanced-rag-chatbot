import streamlit as st

from rag_pipeline import (
    process_documents,
    ask_documents
)

from utils import save_uploaded_file

from config import APP_NAME

# Page config
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    layout="wide"
)

# Load CSS
with open("styles.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Sidebar
with st.sidebar:

    st.title("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=[
            "pdf",
            "txt",
            "csv",
            "html",
            "docx",
            "pptx"
        ],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.info(
        "Upload multiple documents and chat with them."
    )

# Main title
st.title(APP_NAME)

st.markdown(
    "### Advanced Multi-Document RAG System"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Process files
if uploaded_files:

    file_paths = []

    for uploaded_file in uploaded_files:

        file_path = save_uploaded_file(
            uploaded_file
        )

        file_paths.append(file_path)

    with st.spinner("Processing documents..."):

        st.session_state.retriever = (
            process_documents(file_paths)
        )

    st.success("Documents processed successfully!")

# Display history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
question = st.chat_input(
    "Ask questions about your documents..."
)

# Ask question
if question and st.session_state.retriever:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):

        result = ask_documents(
            question,
            st.session_state.retriever
        )

    answer = result["answer"]

    sources = result["sources"]

    final_response = f"""
{answer}

📚 Sources:
{', '.join(sources)}
"""

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response
        }
    )

    with st.chat_message("assistant"):
        st.write(final_response)

elif question and not st.session_state.retriever:

    st.warning("Please upload documents first.")