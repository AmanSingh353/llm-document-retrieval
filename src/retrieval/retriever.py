import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

class Retriever:
    def __init__(self, documents):
        """Initialize the retriever with documents"""
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract text content from documents
        self.texts = []
        self.metadata = []
        
        for doc in documents:
            if hasattr(doc, 'page_content'):
                self.texts.append(doc.page_content)
                self.metadata.append(getattr(doc, 'metadata', {}))
            elif hasattr(doc, 'content'):
                self.texts.append(doc.content)
                self.metadata.append(getattr(doc, 'metadata', {}))
            elif isinstance(doc, dict):
                self.texts.append(doc.get('content', doc.get('page_content', str(doc))))
                self.metadata.append(doc.get('metadata', {}))
            else:
                self.texts.append(str(doc))
                self.metadata.append({})
        
        # Create embeddings and FAISS index
        if self.texts:
            self.embeddings = self.model.encode(self.texts)
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # Inner product for similarity
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        else:
            self.embeddings = None
            self.index = None

    def retrieve(self, query: str, k: int = 5, similarity_threshold: float = 0.4, filters: Optional[Dict[str, Any]] = None):
        """
        Retrieve relevant chunks with optional similarity threshold and metadata filters.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve (default: 5)
            similarity_threshold: Minimum similarity score (default: 0.4)
            filters: Optional metadata filters (default: None)
        
        Returns:
            List of relevant document chunks
        """
        if not self.texts or self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks (get more than k to allow for filtering)
        search_k = min(k * 3, len(self.texts))  # Get 3x more results for filtering
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
                
            # Check similarity threshold
            if score < similarity_threshold:
                continue
            
            # Apply metadata filters if provided
            if filters and self.metadata[idx]:
                metadata = self.metadata[idx]
                
                # File name filter (partial match, case-insensitive)
                if filters.get('file_name'):
                    file_name = metadata.get('file_name', metadata.get('source', ''))
                    if filters['file_name'].lower() not in file_name.lower():
                        continue
                
                # User role filter (exact match)
                if filters.get('user_role'):
                    if metadata.get('user_role', '') != filters['user_role']:
                        continue
                
                # Date range filters
                if filters.get('date_from') or filters.get('date_to'):
                    doc_date = metadata.get('date', metadata.get('created_date'))
                    if doc_date:
                        try:
                            if isinstance(doc_date, str):
                                from datetime import datetime
                                doc_date = datetime.fromisoformat(doc_date.replace('Z', '+00:00')).date()
                            
                            if filters.get('date_from') and doc_date < filters['date_from']:
                                continue
                            if filters.get('date_to') and doc_date > filters['date_to']:
                                continue
                        except (ValueError, TypeError):
                            # Skip date filtering if date format is invalid
                            pass
            
            # Create document chunk object
            chunk = DocumentChunk(
                content=self.texts[idx],
                metadata=self.metadata[idx],
                score=float(score)
            )
            results.append(chunk)
            
            # Stop when we have enough results
            if len(results) >= k:
                break
        
        return results

class DocumentChunk:
    """Simple document chunk class"""
    def __init__(self, content: str, metadata: Dict[str, Any] = None, score: float = 0.0):
        self.content = content
        self.metadata = metadata or {}
        self.score = score
    
    def __str__(self):
        return self.content
    
    def __repr__(self):
        return f"DocumentChunk(content='{self.content[:50]}...', score={self.score:.3f})"
