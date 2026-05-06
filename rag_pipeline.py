import os

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama
)

from loaders import load_document

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)

# Models
llm = ChatOllama(model=MODEL_NAME)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# Process documents
def process_documents(file_paths):

    all_docs = []

    for file_path in file_paths:

        documents = load_document(file_path)

        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)

        all_docs.extend(documents)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = text_splitter.split_documents(all_docs)

    # Vector DB
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    # Save vectorstore
    vectorstore.save_local("vectorstore")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )

    return retriever

# Ask questions
def ask_documents(question, retriever):

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    sources = list(
        set(
            [
                doc.metadata.get("source", "Unknown")
                for doc in retrieved_docs
            ]
        )
    )

    prompt = f"""
    You are an intelligent AI assistant.

    Use ONLY the provided context.

    If information is not available,
    say:
    'The information is not available in the uploaded documents.'

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources
    }