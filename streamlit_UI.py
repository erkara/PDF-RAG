import os
import shutil
import streamlit as st
from rag_main import (
    load_pdfs_from_directory,
    convert_pdfs_chunks,
    create_and_save_vector_store,
    load_vector_store,
    retrieve_context,
    load_generation_model,
    generate_answer,
    get_tokens    # this token is for demonstration
)

# Initialize tokens and upload directory
hf_token = get_tokens()
UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_query(files, query, chunk_size, overlap, embedding_model, generation_model, max_chunks, max_new_tokens, temperature):
    """
    Process the query using uploaded PDFs and return the answer with retrieved sources.
    """
    user_dir = "./user_sessions/user_123"
    vectorstore_path = os.path.join(user_dir, "vectorstore.faiss")
    upload_dir = os.path.join(user_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # Save uploaded PDFs to the session directory
        for uploaded_file in files:
            destination_path = os.path.join(upload_dir, uploaded_file.name)
            with open(destination_path, "wb") as f:
                f.write(uploaded_file.read())

        # Process PDFs and update the vector store
        pdf_documents = load_pdfs_from_directory(upload_dir)
        chunks = convert_pdfs_chunks(pdf_documents, chunk_size=chunk_size, chunk_overlap=overlap)

        if os.path.exists(vectorstore_path):
            # Load and update the existing vector store
            vectorstore = load_vector_store(save_path=vectorstore_path, embedding_model_name=embedding_model)
            vectorstore.add_documents(chunks)
            vectorstore.save_local(vectorstore_path)
        else:
            # Create a new vector store
            vectorstore = create_and_save_vector_store(chunks, embedding_model_name=embedding_model, save_path=vectorstore_path)

        # Retrieve context and generate the answer
        retrieved_chunks, source_info = retrieve_context(vectorstore, query, max_chunks=max_chunks)
        tokenizer, model = load_generation_model(model_name=generation_model)
        answer = generate_answer(tokenizer, model, retrieved_chunks, query, max_new_tokens, temperature)

        # Format retrieved sources for display
        source_info_str = "\n".join([
            f"{source}: {info[0]} chunks, pages {sorted(info[1])}" for source, info in source_info.items()
        ])
        return answer, source_info_str
    except Exception as e:
        return str(e), ""

def main():
    """Streamlit app main function"""
    st.title("PDF Query Assistant")
    st.write("Upload your PDFs, configure parameters, and ask questions about the content.")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    # Input fields
    query = st.text_input("Ask a question about the uploaded PDFs")
    chunk_size = st.slider("Chunk Size", 100, 1000, value=500, step=50)
    overlap = st.slider("Chunk Overlap", 0, 200, value=50, step=10)
    embedding_model = st.selectbox(
        "Embedding Model",
        ["FinLang/finance-embeddings-investopedia", "sentence-transformers/sentence-t5-base"],
        index=0
    )
    generation_model = st.text_input("Generation Model", value="meta-llama/Llama-2-7b-chat-hf")
    max_chunks = st.slider("Max Chunks to Retrieve", 5, 50, value=20, step=1)
    max_new_tokens = st.slider("Max New Tokens", 100, 1000, value=300, step=50)
    temperature = st.slider("Temperature", 0.0, 1.0, value=0.7, step=0.1)

    # Submit button
    if st.button("Generate Answer"):
        if uploaded_files and query:
            answer, sources = process_query(
                uploaded_files, query, chunk_size, overlap, embedding_model,
                generation_model, max_chunks, max_new_tokens, temperature
            )
            st.subheader("Generated Answer")
            st.text(answer)

            st.subheader("Retrieved Sources")
            st.text(sources)
        else:
            st.warning("Please upload files and enter a query to proceed.")

if __name__ == "__main__":
    main()

