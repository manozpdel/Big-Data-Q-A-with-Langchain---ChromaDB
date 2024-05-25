import os
import shutil
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "bigdata"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def main():
    st.title("Big Data Question Answering with Langchain and Streamlit")

    if "responses" not in st.session_state:
        st.session_state.responses = []

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    # Display the latest responses in reverse order
    reversed_responses = reversed(st.session_state.responses)
    for i, (query, response, sources) in enumerate(reversed_responses):
        st.write(f"**Query:** {query}")  # Remove numbering from query
        st.text_area(f"Response", response, height=200, key=f"response_{i}")  # Remove numbering from response
        # st.write(f"Sources: {sources}")

    with st.form(key="query_form"):
        query_text = st.text_input("Enter your prompt:", key="query_input")
        submit_button = st.form_submit_button(label="Submit Query")

        if submit_button and query_text:
            response_text, sources = query_rag(query_text)
            st.session_state.responses.insert(0, (query_text, response_text, sources))  # Insert the latest response at the beginning
            st.session_state.query_text = ""  # Clear the input field after submitting the query
            st.experimental_rerun()  # Rerun to clear the input field and update the UI

def process_documents():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    st.write(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        st.write(f" Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        st.write("No new documents to add")

def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources

if __name__ == "__main__":
    main()
