import os
import sys
import tempfile
from dotenv import load_dotenv
import streamlit as st

# 1)  Make sure project root is on sys.path
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# 2)  Environment variables
# --------------------------------------------------
load_dotenv()                                  # reads .env if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(
        "Environment variable OPENAI_API_KEY is missing. "
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
from src.utils.helpers import format_response

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
                llm = LLMInterface(openai_api_key=OPENAI_API_KEY)

                # ---- c) Run retrieval ----
                parsed_query = QueryInput(raw_query=query)
                relevant_chunks = retriever.retrieve(parsed_query.raw_query)

                if not relevant_chunks:
                    st.info("No relevant content found for this query.")
                    st.stop()

                # ---- d) Call LLM ----
                response = llm.process_query(parsed_query, relevant_chunks)

            # ---- e) Show results ----
            st.subheader("✅ Structured Response")
            st.json(format_response(response))

            st.subheader("📄 Retrieved Clauses")
            for chunk in relevant_chunks:
                st.code(chunk.content.strip(), language="markdown")

        except Exception as err:
            st.error(f"❌ An unexpected error occurred:\n{err}")
