# src/retrieval/retriever.py

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..types.index import DocumentChunk


class Retriever:
    def __init__(self, documents):
        self.embedding_model = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chunks = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(self.chunks, self.embedding_model)

    def retrieve(self, query: str) -> list[DocumentChunk]:
        results = self.vector_store.similarity_search(query, k=5)
        return [
            DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata  # includes source, page number etc.
            )
            for doc in results
        ]
