import os
import requests
from typing import List
from ..types.index import QueryInput, DocumentChunk

class LLMInterface:
    def __init__(self, perplexity_api_key=None):
        """Initialize Perplexity API client"""
        self.api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
        
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY is required")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def process_query(self, parsed_query: QueryInput, chunks: List[DocumentChunk]):
        """Process query using Perplexity API with safe JSON handling"""
        
        if not chunks:
            return {
                "answer": "No matching information found in the provided documents.",
                "justification": "No relevant content was retrieved from the uploaded documents."
            }
        
        # Combine chunk content safely
        context_parts = []
        for chunk in chunks:
            if hasattr(chunk, 'content'):
                context_parts.append(chunk.content)
            elif isinstance(chunk, dict):
                content = chunk.get('content', chunk.get('page_content', str(chunk)))
                context_parts.append(content)
            else:
                context_parts.append(str(chunk))
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following document content, answer the user's query:

CONTEXT:
{context[:4000]}

QUERY: {parsed_query.raw_query}

Answer based only on the provided information."""
        
        # Updated payload with current model
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based only on provided document content."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                headers=self.headers,
                timeout=30
            )
            
            # Check response status first
            if response.status_code != 200:
                return {
                    "answer": f"API Error: {response.status_code} - {response.text}",
                    "justification": "Perplexity API call failed"
                }
            
            # Safely handle JSON parsing
            if not response.text.strip():
                return {
                    "answer": "Empty response from API",
                    "justification": "API returned empty response"
                }
            
            try:
                result = response.json()
            except ValueError:
                # JSON decode error - return the raw text
                return {
                    "answer": f"Invalid JSON response: {response.text}",
                    "justification": "API returned invalid JSON format"
                }
            
            # Extract answer from valid JSON response
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                return {
                    "answer": answer,
                    "justification": f"Based on {len(chunks)} relevant document sections."
                }
            else:
                return {
                    "answer": "No response generated",
                    "justification": "Empty response from API"
                }
            
        except requests.exceptions.RequestException as e:
            return {
                "answer": f"Network error: {str(e)}",
                "justification": "API connection failed"
            }
        except Exception as e:
            return {
                "answer": f"Unexpected error: {str(e)}",
                "justification": "Processing failed"
            }
