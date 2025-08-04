# 🦜🔗 LLM Document Retrieval System

Ask natural-language questions about your **policy, contract, or any legal document** and get instant, cited answers powered by a local vector store and a Large Language Model (LLM).

<img src="assets/screenshot.png" width="800" alt="App screenshot" />

---

## ✨ Features
- **Multi-format ingestion** – PDF, DOCX, TXT, EML (200 MB per file)  
- **Automatic chunking & embeddings** – creates a FAISS index for fast similarity search  
- **Retrieval-Augmented Generation (RAG)** – feeds relevant chunks to the LLM for grounded answers  
- **Streamlit UI** – drag-and-drop uploads, chat-style Q&A, source citations  
- **Audit logging** – every query, answer, and document ID is stored in JSONL  
- **Pluggable models** – switch between free HuggingFace embeddings or OpenAI embeddings with one env flag  
- **Test suite** – unit, integration, performance, and quality checks under `tests/`

---

## 🗂️ Project Structure
