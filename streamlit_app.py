import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables (e.g. API keys)
load_dotenv()

# --- Imports from local modules ---
from src.data.loader import load_data
from src.retrieval.retriever import Retriever
from src.llm.llm_interface import LLMInterface
from src.types.index import QueryInput
from src.utils.helpers import format_response


# --- Streamlit page config ---
st.set_page_config(page_title="📄 LLM Document Query", layout="wide")
st.title("📄 LLM Document Retrieval System")
st.markdown("Ask questions about your policy, contract, or legal documents using AI.")

# --- User Inputs ---
query = st.text_input("Enter your query:", placeholder="e.g. 46-year-old male, knee surgery in Pune, 3-month-old policy")
uploaded_files = st.file_uploader("Upload document(s)", type=["pdf", "docx", "txt", "eml"], accept_multiple_files=True)

# --- Run Button ---
if st.button("🔍 Run Query"):
    if not query.strip():
        st.warning("⚠ Please enter a valid query.")
    elif not uploaded_files:
        st.warning("⚠ Please upload at least one document.")
    else:
        with st.spinner("🔄 Processing your request..."):
            try:
                # --- Save uploaded files to a temporary directory ---
                with tempfile.TemporaryDirectory() as tmp_dir:
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(tmp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())
                        file_paths.append(file_path)

                    # --- Load documents via loader ---
                    documents = load_data(tmp_dir)

                    # --- Initialize Retrieval & LLM Components ---
                    retriever = Retriever(documents)
                    llm = LLMInterface()

                    # --- Form structured query ---
                    parsed_query = QueryInput(raw_query=query)

                    # --- Retrieve relevant content ---
                    relevant_chunks = retriever.retrieve(parsed_query.raw_query)

                    # --- LLM inference step ---
                    response = llm.process_query(parsed_query, relevant_chunks)

                    # --- Display Outputs ---
                    st.subheader("✅ Structured Response")
                    st.json(format_response(response))

                    st.subheader("📄 Retrieved Clauses")
                    if relevant_chunks:
                        for chunk in relevant_chunks:
                            st.code(chunk.content.strip(), language="markdown")
                    else:
                        st.info("No relevant content retrieved from the documents.")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")