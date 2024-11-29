#!/usr/bin/env python
# coding: utf-8



# Initialization
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from functools import lru_cache
import os
from collections import Counter, defaultdict


"""Load environment variables and configure device."""
load_dotenv("keys.txt")
hf_token = os.getenv("HF_TOKEN")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")

def get_tokens(key_file_path='keys.txt'):
    load_dotenv(key_file_path)
    hf_token = os.getenv("HF_TOKEN")

    return hf_token


embedding_model_name = 'FinLang/finance-embeddings-investopedia'
def load_pdfs_from_directory(directory_path):
    """
    Load all PDFs from a directory.
    Each PDF is treated as a separate collection of pages.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        dict: A dictionary where keys are filenames and values are lists of Langchain "document" objects(pages)
    """
    pdf_documents = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pdf"):  # Only process PDF files TODO: generalize
            file_path = os.path.join(directory_path, file_name)
            print(f"Loading: {file_name}")
            
            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # file names are the keys,i.e,
            pdf_documents[file_name] = documents
    print(f"Loaded {len(pdf_documents)} PDFs from {directory_path}")
    return pdf_documents


def convert_pdfs_chunks(pdf_documents, chunk_size=500, chunk_overlap=50):
    """
    Split the content of each PDF into smaller chunks, preserving metadata.

    Args:
        pdf_documents (dict): A dictionary where keys are PDF filenames and values are lists of page documents.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: A list of chunked documents, each with metadata pointing to its source PDF and page.
        this will be imporant later on during retrival evaluation.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for file_name, documents in pdf_documents.items():
        print(f"Splitting pages from {file_name} into chunks...")
        for document in documents:
            # Extract the page number from the document metadata
            page_number = document.metadata.get("page", "unknown")
            if page_number != "unknown":
                page_number += 1

            # Split the document into chunks
            chunks = text_splitter.split_documents([document])

            # Add metadata (source and page) to each chunk
            for chunk in chunks:
                chunk.metadata = {
                    "source": file_name,
                    "page": page_number
                }

            # get'em all
            all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks



def create_and_save_vector_store(chunks, embedding_model_name='sentence-transformers/sentence-t5-large',
                                 save_path="vectorstore.faiss"):
    """
    Embed the chunks and save a vector store to disk.
    
    Args:
        chunks (list): List of chunked documents with metadata.
        embedding_model_name (str): Name of the Hugging Face embedding model.
        save_path (str): Path to save the vector store.

    Returns:
        FAISS: The created vector store.
    """
    print("Generating embeddings for chunks...")
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create the vector store
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(save_path)
    
    print(f"Vector store saved to {save_path}")
    return vectorstore
    

def load_vector_store(embedding_model_name, save_path="vectorstore.faiss"):
    """
    Load a saved FAISS vector store from disk.
    """
    print(f"Loading vector store from {save_path}...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)


def retrieve_context(vectorstore, query, max_chunks=10, show_source=False):
    """
    Retrieve relevant chunks from the vector store based on a query and provide source counts with pages.

    Args:
        vectorstore (FAISS): The vector store object.
        query (str): The user's query.
        max_chunks (int): Maximum number of chunks to retrieve.

    Returns:
        tuple:
            - List of retrieved chunks, each containing content and metadata.
            - Dictionary with source counts and pages in the format:
              {'source.pdf': [chunk_count, [page_numbers]]}
    """
    print(f"Querying vector store for: '{query}'")
    retrieved_docs = vectorstore.similarity_search(query, k=max_chunks)

    # Initialize a dictionary to store source counts and pages
    source_info = {}

    # Populate the source counts and pages
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        if source not in source_info:
            source_info[source] = [0, []]  # Initialize count and page list

        # Increment the count and append the page (if unique)
        source_info[source][0] += 1
        if page != "unknown" and page not in source_info[source][1]:
            source_info[source][1].append(page)

    # Print the source info
    if show_source:
        print("==========================Source Documents/Respective Pages=======================\n", source_info)
    return retrieved_docs, source_info


@lru_cache(maxsize=1)  # Cache the model and tokenizer
def load_generation_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Load a quantized Llama model with cached loading.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print(f"Loading generation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token
    )
    return tokenizer, model



def generate_answer(tokenizer, model, retrieved_chunks, query, max_new_tokens=200,temperature=0.7):
    """
    Generate an answer based on retrieved chunks and a query.
    
    Args:
        tokenizer: Tokenizer for the model.
        model: The generation model.
        retrieved_chunks (list): List of retrieved text chunks.
        query (str): User's query.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated answer.
    """
    # Combine retrieved chunks into a single context
    context = " ".join([chunk.page_content for chunk in retrieved_chunks])
    
    # Construct the input for the model
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = inputs.to(model.device)  # Move to the same device as the model
    
    # Generate the response
    outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer portion
    return response.split("Answer:")[-1].strip()




def format_references_and_pages(model_references, model_pages):
    # Group pages by their corresponding source
    grouped_data = defaultdict(list)
    for ref, page in zip(model_references, model_pages):
        grouped_data[ref].append(page)

    # Format Model References (unique, sorted list)
    formatted_references = ", ".join(sorted(set(model_references)))

    # Format Pages as detailed key-value pairs
    formatted_pages = "; ".join(
        f"{source}: {sorted(pages)}" for source, pages in grouped_data.items()
    )

    return formatted_references, formatted_pages
