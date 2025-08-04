# src/main.py

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env before anything else

from retrieval.retriever import Retriever
from llm.llm_interface import LLMInterface
from data.loader import load_data
from utils.helpers import format_response
from types.index import QueryInput

def main():
    query_str = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    query = QueryInput(raw_query=query_str)

    documents = load_data("./documents")

    retriever = Retriever(documents)
    llm = LLMInterface()

    relevant_chunks = retriever.retrieve(query.raw_query)
    response = llm.process_query(query, relevant_chunks)

    print(format_response(response))

if __name__ == "__main__":
    main()
