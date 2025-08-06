# src/types/index.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class QueryInput:
    raw_query: str

@dataclass 
class DocumentChunk:
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

