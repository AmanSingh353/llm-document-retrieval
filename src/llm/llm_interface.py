# src/llm/llm_interface.py

from typing import List
from openai import OpenAI
from ..types.index import QueryInput, DocumentChunk, LLMResponse
import os

class LLMInterface:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def process_query(self, query: QueryInput, chunks: List[DocumentChunk]) -> LLMResponse:
        content_blocks = []
        for chunk in chunks:
            header = f"[Source: {chunk.metadata.get('source', 'unknown')}]"
            body = chunk.content.strip()
            content_blocks.append(f"{header}\n{body}")

        context = "\n\n".join(content_blocks)

        prompt = f"""
You are an insurance assistant. Analyze the following policy document sections:

{context}

Given this user query:
"{query.raw_query}"

Return a JSON object with:
- decision: "Approved" or "Rejected"
- amount: payout amount if approved, otherwise "N/A"
- justification: explain your decision
- clauses: list of clause numbers or section headers you based your decision on (e.g., ["5.2(b)", "3.1"])

Respond only with JSON.
"""

        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        reply = completion.choices[0].message.content.strip()
        return LLMResponse.from_json(reply)
