import os
import sys
import tempfile
from dotenv import load_dotenv
import streamlit as st
import traceback
import json

# 1)  Make sure project root is on sys.path
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# 2)  Environment variables
# --------------------------------------------------
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not PERPLEXITY_API_KEY:
    st.error(
        "Environment variable PERPLEXITY_API_KEY is missing. "
        "Add it to your system env or a .env file."
    )
    st.stop()

# --------------------------------------------------
# 3)  Local imports (src.* assumes folder layout:  src/...)
# --------------------------------------------------
from src.data.loader import load_data
from src.retrieval.retriever import Retriever
from src.llm.llm_interface import LLMInterface
from src.types.index import QueryInput

# --------------------------------------------------
# 4)  Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="📄 LLM Document Query", layout="wide")
st.title("📄 LLM Document Retrieval System")
st.markdown("Ask questions about your policy, contract, or legal documents using AI.")

query = st.text_input(
    "Enter your query:",
    placeholder="e.g. 46-year-old male, knee surgery in Pune, 3-month-old policy",
)

uploaded_files = st.file_uploader(
    "Upload document(s)",
    type=["pdf", "docx", "txt", "eml"],
    accept_multiple_files=True,
)

# --------------------------------------------------
# 5)  Button click handler
# --------------------------------------------------
if st.button("🔍 Run Query"):
    if not query.strip():
        st.warning("⚠ Please enter a valid query.")
        st.stop()

    if not uploaded_files:
        st.warning("⚠ Please upload at least one document.")
        st.stop()

    with st.spinner("🔄 Processing your request..."):
        try:
            # ---- a) Save uploads to a temp dir ----
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uf in uploaded_files:
                    dst = os.path.join(tmp_dir, uf.name)
                    with open(dst, "wb") as fp:
                        fp.write(uf.read())

                # ---- b) Load docs and build retriever ----
                documents = load_data(tmp_dir)
                retriever = Retriever(documents)
                llm = LLMInterface(perplexity_api_key=PERPLEXITY_API_KEY)

                # ---- c) Run retrieval with error handling ----
                parsed_query = QueryInput(raw_query=query)
                
                try:
                    relevant_chunks = retriever.retrieve(parsed_query.raw_query)
                except Exception as e:
                    st.error("❌ Error during document retrieval")
                    st.code(traceback.format_exc(), language="python")
                    st.stop()

                if not relevant_chunks:
                    st.info("No relevant content found for this query.")
                    st.stop()

                # ---- d) Call LLM with error handling ----
                try:
                    response = llm.process_query(parsed_query, relevant_chunks)
                except Exception as e:
                    st.error("❌ Error during LLM processing")
                    st.code(traceback.format_exc(), language="python")
                    st.stop()

            # ---- e) Show results (UPDATED - bypass format_response) ----
            st.subheader("✅ Structured Response")
            
            # Handle response display safely without format_response function
            if isinstance(response, dict):
                # Response is already a dictionary, display directly
                st.json(response)
                
                # Also display answer prominently if available
                if "answer" in response:
                    st.markdown("### 📝 Answer:")
                    st.write(response["answer"])
                    
            elif isinstance(response, str):
                # Try to parse as JSON, fallback to text display
                try:
                    parsed_response = json.loads(response)
                    st.json(parsed_response)
                    if isinstance(parsed_response, dict) and "answer" in parsed_response:
                        st.markdown("### 📝 Answer:")
                        st.write(parsed_response["answer"])
                except json.JSONDecodeError:
                    st.markdown("### 📝 Response:")
                    st.write(response)
            else:
                # Fallback for any other type
                st.markdown("### 📝 Response:")
                st.write(str(response))

            # ---- f) Display retrieved chunks safely ----
            st.subheader("📄 Retrieved Clauses")
            if relevant_chunks:
                for i, chunk in enumerate(relevant_chunks):
                    try:
                        # Handle different chunk formats safely
                        if hasattr(chunk, 'content'):
                            # It's a proper DocumentChunk object
                            content = chunk.content
                        elif isinstance(chunk, dict):
                            # It's a dictionary - extract content
                            content = chunk.get('content', chunk.get('page_content', str(chunk)))
                        else:
                            # Fallback for unexpected formats
                            content = str(chunk)
                        
                        # Make sure content is a string before calling strip()
                        if isinstance(content, str) and content.strip():
                            st.code(content.strip(), language="markdown")
                        else:
                            st.code(str(content), language="text")
                            
                    except Exception as e:
                        st.write(f"⚠️ Could not display chunk {i+1}: {str(e)}")
                        st.code(str(chunk), language="text")
            else:
                st.info("No relevant content retrieved from the documents.")

        except Exception as err:
            st.error("❌ An unexpected error occurred")
            st.code(traceback.format_exc(), language="python")
