# src/types/index.py

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class QueryInput:
    raw_query: str

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict = None  # Added metadata for clause tracking

@dataclass
class LLMResponse:
    decision: str
    amount: str
    justification: str
    clauses: List[str] = None  # List of clause IDs or summaries

    @staticmethod
    def from_json(json_str: str) -> 'LLMResponse':
        import json
        try:
            data = json.loads(json_str)
            return LLMResponse(
                decision=data.get("decision", "Unknown"),
                amount=data.get("amount", "N/A"),
                justification=data.get("justification", "No justification provided."),
                clauses=data.get("clauses", [])
            )
        except json.JSONDecodeError:
            return LLMResponse(
                decision="Error",
                amount="N/A",
                justification="Failed to parse LLM response JSON.",
                clauses=[]
            )
