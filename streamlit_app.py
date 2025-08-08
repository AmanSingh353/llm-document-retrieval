import os
import sys
import tempfile
from dotenv import load_dotenv
import streamlit as st
import traceback
import json
import requests

# 1) Make sure project root is on sys.path
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 2) Environment variables
# --------------------------------------------------
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# API configuration for PDF URL processing
API_URL = "https://your-fastapi-app.onrender.com/api/v1/hackrx/run"
API_KEY = "your_api_key_here"

if not PERPLEXITY_API_KEY:
    st.error(
        "Environment variable PERPLEXITY_API_KEY is missing. "
        "Add it to your system env or a .env file."
    )
    st.stop()

# 3) Local imports
# --------------------------------------------------
try:
    from src.data.loader import load_data
    from src.retrieval.retriever import Retriever
    from src.llm.llm_interface import LLMInterface
    from src.types.index import QueryInput
    RAG_COMPONENTS_LOADED = True
except ImportError as e:
    st.error(f"❌ Could not import RAG components: {e}")
    st.error("Please check your project structure and dependencies.")
    RAG_COMPONENTS_LOADED = False

# --------------------------------------------------
# HELPER FUNCTIONS (MOVE THESE TO THE TOP!)
# --------------------------------------------------
def display_upload_results(response, relevant_chunks, query, k, similarity_threshold, uploaded_files, filters):
    """Display results for upload mode"""
    
    # Main response section
    st.subheader("✅ AI Response")
    
    # Handle response display safely
    if isinstance(response, dict):
        # Display main answer prominently
        if "answer" in response:
            st.markdown("### 📝 Answer:")
            st.write(response["answer"])
            
            # Show justification if available
            if "justification" in response:
                with st.expander("🔍 Response Details"):
                    st.write("**Justification:**", response["justification"])
        
        # Show full JSON response in expander
        with st.expander("🔧 Full Response Data"):
            st.json(response)
            
    elif isinstance(response, str):
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
        st.markdown("### 📝 Response:")
        st.write(str(response))

    # Retrieved chunks section
    st.subheader("📄 Retrieved Document Sections")
    
    if relevant_chunks:
        # Show summary
        st.info(f"📊 Found {len(relevant_chunks)} relevant sections from your documents")
        
        # Filter settings display
        if filters:
            with st.expander("🔍 Applied Filters"):
                for key, value in filters.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Display chunks
        for i, chunk in enumerate(relevant_chunks):
            with st.expander(f"📄 Section {i+1}"):
                try:
                    # Handle different chunk formats safely
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        metadata = getattr(chunk, 'metadata', {})
                    elif isinstance(chunk, dict):
                        content = chunk.get('content', chunk.get('page_content', str(chunk)))
                        metadata = chunk.get('metadata', {})
                    else:
                        content = str(chunk)
                        metadata = {}
                    
                    # Display metadata if available
                    if metadata:
                        st.caption("**Source Information:**")
                        for key, value in metadata.items():
                            if value:  # Only show non-empty values
                                st.caption(f"• **{key.replace('_', ' ').title()}:** {value}")
                    
                    # Display content
                    if isinstance(content, str) and content.strip():
                        st.code(content.strip(), language="markdown")
                    else:
                        st.code(str(content), language="text")
                        
                except Exception as e:
                    st.error(f"⚠️ Could not display section {i+1}: {str(e)}")
                    st.code(str(chunk), language="text")
    else:
        st.info("No relevant content retrieved from the documents.")

    # Query information
    with st.expander("ℹ️ Query Information"):
        st.write("**Query:** ", query)
        st.write("**Number of chunks requested:** ", k)
        st.write("**Similarity threshold:** ", similarity_threshold)
        st.write("**Documents processed:** ", len(uploaded_files))
        if filters:
            st.write("**Filters applied:** ", len(filters))

def display_pdf_results(api_response, question_list, pdf_url):
    """Display results for PDF URL mode"""
    
    st.subheader("✅ PDF Analysis Results")
    
    # Show PDF source
    st.info(f"📄 **Source:** {pdf_url}")
    
    try:
        answers = api_response.get("answers", [])
        
        if not answers:
            st.warning("⚠️ No answers received from the API")
            st.json(api_response)
            return
        
        # Display Q&A pairs
        for i, (question, answer) in enumerate(zip(question_list, answers)):
            st.markdown(f"### ❓ Question {i+1}")
            st.markdown(f"**{question}**")
            
            st.markdown("### 📝 Answer")
            st.write(answer)
            
            if i < len(question_list) - 1:  # Add separator except for last item
                st.markdown("---")
        
        # Show summary
        with st.expander("📊 Summary"):
            st.write(f"**Questions processed:** {len(question_list)}")
            st.write(f"**Answers received:** {len(answers)}")
            st.write(f"**PDF URL:** {pdf_url}")
        
        # Show raw API response
        with st.expander("🔧 Raw API Response"):
            st.json(api_response)
            
    except Exception as e:
        st.error("❌ Error processing API response")
        st.json(api_response)

# --------------------------------------------------
# 4) Streamlit UI Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="📄 LLM Document Query", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 LLM Document Retrieval System")
st.markdown("Ask questions about your documents using AI - Upload files or provide PDF URLs.")

# ... rest of your code continues here ...
