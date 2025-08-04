# src/data/loader.py

import os
from langchain.document_loaders import (
    PyMuPDFLoader,
    UnstructuredEmailLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

def load_data(directory: str):
    documents = []
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)


        try:
            if filename.endswith(".pdf"):
                loader = PyMuPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(filepath)
            elif filename.endswith(".eml"):
                loader = UnstructuredEmailLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
            else:
                print(f"[SKIP] Unsupported file format: {filename}")
                continue

            docs = loader.load()
            documents.extend(docs)

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

    return documents
