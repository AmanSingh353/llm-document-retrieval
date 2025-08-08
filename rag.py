import os
from dotenv import load_dotenv
from src.data.loader import load_data
from src.retrieval.retriever import Retriever
from src.llm.llm_interface import LLMInterface
from src.types.index import QueryInput

def run_rag_pipeline(file_path, question):
    # Load environment variables
    load_dotenv()

    # Load and embed document
    print("🔄 Loading and embedding document...")
    documents = load_data([file_path])

    # Initialize Retriever and LLM
    retriever = Retriever()
    llm = LLMInterface()

    # Create query input
    query = QueryInput(question=question)

    # Run retrieval and generation
    print("🧠 Generating answer using RAG...")
    result = llm.answer(query, retriever)

    print("\n✅ Answer:")
    print(result.answer)

    if result.sources:
        print("\n📚 Sources:")
        for source in result.sources:
            print(f"- {source}")

    return result.answer

if __name__ == "__main__":
    # Sample usage
    file_path = "your_document.pdf"  # Replace with actual path
    question = "Summarize the policy coverage in simple terms."  # Replace with user query
    run_rag_pipeline(file_path, question)
