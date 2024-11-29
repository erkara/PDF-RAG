#!/usr/bin/env python


import atexit
import shutil
import gradio as gr
from rag_main import (
    load_pdfs_from_directory,
    convert_pdfs_chunks,
    create_and_save_vector_store,
    load_vector_store,
    retrieve_context,
    load_generation_model,
    generate_answer,
    get_tokens # this token is for demonstration
)
import os

# Initialize tokens and upload directory
hf_token = get_tokens()
UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)



def process_query(files, query, chunk_size, overlap, embedding_model, generation_model, max_chunks, max_new_tokens, temperature):
    """
    Process the query using uploaded PDFs and return the answer with retrieved sources.
    """
    # Use a static session directory
    user_dir = "./user_sessions/user_123"
    vectorstore_path = os.path.join(user_dir, "vectorstore.faiss")
    upload_dir = os.path.join(user_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # Save new uploaded PDFs to the session directory
        for file_path in files:
            destination_path = os.path.join(upload_dir, os.path.basename(file_path))
            shutil.copy(file_path, destination_path)  # Use copy instead of rename

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



def clean_session_on_exit():
    """
    Deletes all session-related data when the app shuts down.
    """
    session_dir = "./user_sessions/user_123"
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        print("Session data deleted on app shutdown.")

# Register the cleanup function
atexit.register(clean_session_on_exit)



# Gradio Interface TODO: add descriptions as well.
interface = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload PDFs", file_types=[".pdf"], type="filepath", file_count="multiple"),
        gr.Textbox(placeholder="Ask a question about the uploaded PDFs", label="Query"),
        gr.Slider(100, 1000, value=500, step=50, label="Chunk Size"),
        gr.Slider(0, 200, value=50, step=10, label="Chunk Overlap"),
        gr.Dropdown(
            choices=["FinLang/finance-embeddings-investopedia", "sentence-transformers/sentence-t5-base"],
            value="FinLang/finance-embeddings-investopedia",
            label="Embedding Model",
        ),
        gr.Textbox(value="meta-llama/Llama-2-7b-chat-hf", label="Generation Model"),
        gr.Slider(5, 50, value=20, step=1, label="Max Chunks to Retrieve"),
        gr.Slider(100, 1000, value=300, step=50, label="Max New Tokens"),
        gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Retrieved Sources"),
    ],
    title="PDF Query Assistant",
    description="Upload your PDFs, configure parameters, and ask questions about the content."
)

# Register cleanup on shutdown
atexit.register(clean_session_on_exit)

# Launch the app in Jupyter
interface.launch(share=True, inline=False)





