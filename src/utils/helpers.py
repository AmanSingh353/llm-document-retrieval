# src/utils/helpers.py

import datetime
import json
from typing import List, Dict, Optional
import os


def format_response(answer: str) -> str:
    """
    Format LLM answer for output. You can extend this with markdown or styling.
    """
    return f"### ✅ Response\n\n{answer.strip()}"


def log_query(
    user_query: str,
    relevant_docs: List[Dict],
    answer: str,
    username: Optional[str] = "anonymous",
    log_file: str = "logs/audit_log.jsonl"
):
    """
    Logs the user query, relevant document IDs, and the final LLM answer.
    Saved in JSONL format for easy audit and parsing.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": username,
        "query": user_query,
        "docs_considered": [doc.get("id", "?") for doc in relevant_docs],
        "response": {
            "decision": extract_decision(answer),
            "amount": extract_amount(answer),
            "justification": answer.strip(),
            "clauses": [doc.get("clause_id", "?") for doc in relevant_docs if "clause_id" in doc]
        }
    }

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def extract_decision(text: str) -> str:
    """
    Dummy logic to extract decision from the LLM answer.
    Customize this based on how your LLM answers are structured.
    """
    text_lower = text.lower()
    if "approved" in text_lower:
        return "Approved"
    elif "rejected" in text_lower or "not covered" in text_lower:
        return "Rejected"
    return "Unknown"


def extract_amount(text: str) -> Optional[str]:
    """
    Dummy logic to extract amount (if mentioned) from LLM answer.
    """
    import re
    match = re.search(r"(?:₹|\$|Rs\.?)\s?\d+(?:,\d{3})*", text)
    return match.group(0) if match else None
