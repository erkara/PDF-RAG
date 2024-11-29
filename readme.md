# PDF-RAG: Retrieval-Augmented Generation for PDFs

Welcome to **PDF-RAG**, a simple pipeline that lets you upload and interact with your PDFs. This repository provides an easy-to-use framework for building a conversational interface for document interaction.

## Getting Started

1. **Set up your environment:**

   ```bash
   python3 -m venv my_new_env
   source my_new_env/bin/activate
   ```

2. **Follow the notebook:**
   Open `rag_demo.ipynb` for step-by-step instructions. Sample documents from the Congressional Budget Office are included in the `cbo_documents` folder for testing.

## Pipeline Overview

- **Processing PDFs:** PDFs are chunked into manageable pieces using LangChain.
- **Embeddings:**
  - We tested **[FinLang](https://huggingface.co/FinLang/finance-embeddings-investopedia)** (for financial documents) and **[sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base)** (for general use).
  - Embeddings are stored in **Faiss** for similarity search.
- **Generation:** **[Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)** powers the conversational interface.

## User Interfaces

Interact with your PDFs using:

- **Streamlit:**
  ```bash
  streamlit run streamlit_UI.py
  ```
- **Gradio:**
  ```bash
  python gradio_UI.py
  ```

## Notes

The pipeline works great for text-heavy documents but isn’t the best fit for those with complex multi-modal content just yet. Don’t worry—we’re cooking up an update to tackle that with a multi-modal RAG pipeline powered by Vision LLMs. Until then, we’d love to hear your feedback or questions. Cheers!
