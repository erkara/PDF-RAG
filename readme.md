# PDF-RAG: Retrieval-Augmented Generation for PDFs

Welcome to **PDF-RAG**, a simple pipeline that lets you upload and interact with your PDFs. This repository provides an easy-to-use framework for building a conversational interface for document interaction.

## Getting Started

1. **Set up your environment:**

   ```bash
   python3 -m venv my_new_env
   source my_new_env/bin/activate
   ```

2. **Follow the notebook:**
   Open `rag_demo.ipynb` for step-by-step instructions. We’ve also included a test set of questions, answers, and source documents in cbo_questions.xlsx to evaluate the pipeline. Sample documents from the Congressional Budget Office are included in the cbo_documents folder for testing. *To make this work, please go to "keys.txt" file and paste your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to access the models*.  

## Pipeline Overview

- **Processing PDFs:** PDFs are chunked into manageable pieces using LangChain.
- **Embeddings:**
  - We tested **[FinLang](https://huggingface.co/FinLang/finance-embeddings-investopedia)** (for financial documents) and **[sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base)** (for general use).
  - Embeddings are managed using [**Faiss**](https://github.com/facebookresearch/faiss), which is optimized for fast similarity searches.
- **Generation:** **[Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)** powers the conversational interface.

## User Interfaces

Interact with your PDFs using the following code snippets. You can also try out on [Google Colab](https://colab.research.google.com/drive/1Nx5bLktqCXLg_wRYgz7_DlidxaF8Cmo9?usp=drive_link)

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
