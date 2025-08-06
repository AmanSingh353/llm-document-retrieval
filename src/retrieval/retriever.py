from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..types.index import DocumentChunk

class Retriever:
    def __init__(self, documents):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chunks = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(self.chunks, self.embedding_model)

    def retrieve(self, query: str) -> list[DocumentChunk]:
        results = self.vector_store.similarity_search(query, k=5)
        
        if not results:
            return []
        
        # Convert LangChain Documents to DocumentChunk objects
        document_chunks = []
        for doc in results:
            chunk = DocumentChunk(
                content=doc.page_content,  # LangChain uses 'page_content', not 'content'
                metadata=doc.metadata if doc.metadata else {}
            )
            document_chunks.append(chunk)
            
        return document_chunks
