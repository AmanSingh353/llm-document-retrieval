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
try:
    from src.data.loader import load_data
    from src.retrieval.retriever import Retriever
    from src.llm.llm_interface import LLMInterface
    from src.types.index import QueryInput
    RAG_COMPONENTS_LOADED = True
except ImportError as e:
    st.error(f"❌ Could not import RAG components: {e}")
    st.error("Please check your project structure and dependencies.")
    st.stop()

# --------------------------------------------------
# 4)  Streamlit UI Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="📄 LLM Document Query", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 LLM Document Retrieval System")
st.markdown("Ask questions about your policy, contract, or legal documents using AI.")

# --------------------------------------------------
# 5)  Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Retrieval Settings")
    
    # Retrieval parameters
    k = st.slider(
        "Number of top chunks (k)", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Number of most relevant document chunks to retrieve"
    )
    
    similarity_threshold = st.slider(
        "Similarity threshold", 
        min_value=0.0, 
        max_value=1.0, 
        step=0.01, 
        value=0.4,
        help="Minimum similarity score for chunk relevance"
    )
    
    st.markdown("## 🔍 Optional Filters")
    
    # Metadata filters
    file_name_filter = st.text_input(
        "File name contains...", 
        value="",
        help="Filter chunks by filename (partial match)"
    )
    
    user_role_filter = st.text_input(
        "User role (exact match)", 
        value="",
        help="Filter by user role metadata"
    )
    
    # Date range filters
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input(
            "Date from", 
            value=None,
            help="Filter documents from this date"
        )
    with col2:
        date_to = st.date_input(
            "Date to", 
            value=None,
            help="Filter documents up to this date"
        )
    
    st.markdown("## 📊 Model Settings")
    
    # Model temperature (if available)
    temperature = st.slider(
        "Response creativity", 
        min_value=0.0, 
        max_value=1.0, 
        step=0.1, 
        value=0.2,
        help="Higher values make responses more creative"
    )
    
    max_tokens = st.number_input(
        "Max response length", 
        min_value=100, 
        max_value=2000, 
        value=500,
        help="Maximum number of tokens in response"
    )

# --------------------------------------------------
# 6)  Main Interface
# --------------------------------------------------
query = st.text_input(
    "Enter your query:",
    placeholder="e.g. What is the grace period for premium payments?",
    help="Ask specific questions about your uploaded documents"
)

uploaded_files = st.file_uploader(
    "Upload document(s)",
    type=["pdf", "docx", "txt", "eml"],
    accept_multiple_files=True,
    help="Upload one or more documents to query"
)

# Display uploaded files info
if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded: {', '.join([f.name for f in uploaded_files])}")

# --------------------------------------------------
# 7)  Processing Section
# --------------------------------------------------
if st.button("🔍 Run Query", type="primary"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid query.")
        st.stop()

    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document.")
        st.stop()

    # Prepare filters
    filters = {
        "file_name": file_name_filter.strip() or None,
        "user_role": user_role_filter.strip() or None,
        "date_from": date_from,
        "date_to": date_to
    }
    
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    # Processing with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🔄 Processing your request..."):
        try:
            # Step 1: Save uploaded files
            status_text.text("📁 Saving uploaded files...")
            progress_bar.progress(10)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uf in uploaded_files:
                    dst = os.path.join(tmp_dir, uf.name)
                    with open(dst, "wb") as fp:
                        fp.write(uf.read())

                # Step 2: Load and process documents
                status_text.text("📖 Loading and processing documents...")
                progress_bar.progress(30)
                
                documents = load_data(tmp_dir)
                
                if not documents:
                    st.error("❌ No content could be extracted from uploaded files.")
                    st.stop()

                # Step 3: Initialize retrieval system
                status_text.text("🔧 Initializing retrieval system...")
                progress_bar.progress(50)
                
                retriever = Retriever(documents)
                llm = LLMInterface(perplexity_api_key=PERPLEXITY_API_KEY)

                # Step 4: Retrieve relevant chunks
                status_text.text("🔍 Retrieving relevant content...")
                progress_bar.progress(70)
                
                parsed_query = QueryInput(raw_query=query)
                
                try:
                    # Check if retriever supports advanced parameters
                    if hasattr(retriever, 'retrieve') and len(retriever.retrieve.__code__.co_varnames) > 2:
                        relevant_chunks = retriever.retrieve(
                            parsed_query.raw_query,
                            k=k,
                            similarity_threshold=similarity_threshold,
                            filters=filters if filters else None
                        )
                    else:
                        # Fallback to basic retrieval
                        relevant_chunks = retriever.retrieve(parsed_query.raw_query)
                        
                except Exception as e:
                    st.error("❌ Error during document retrieval")
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc(), language="python")
                    st.stop()

                if not relevant_chunks:
                    st.info("🔍 No relevant content found for this query.")
                    st.info("💡 Try adjusting the similarity threshold or using different keywords.")
                    st.stop()

                # Step 5: Generate response
                status_text.text("🤖 Generating AI response...")
                progress_bar.progress(90)
                
                try:
                    response = llm.process_query(parsed_query, relevant_chunks)
                except Exception as e:
                    st.error("❌ Error during LLM processing")
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc(), language="python")
                    st.stop()

            # Step 6: Complete processing
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # --------------------------------------------------
            # 8)  Results Display
            # --------------------------------------------------
            
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

        except Exception as err:
            st.error("❌ An unexpected error occurred")
            
            # Enhanced error display
            with st.expander("🐛 Error Details"):
                st.code(traceback.format_exc(), language="python")
                
                # Helpful error suggestions
                st.markdown("### 💡 Possible Solutions:")
                st.markdown("""
                - Check that all required dependencies are installed
                - Verify your `.env` file contains the correct API keys
                - Ensure uploaded files are not corrupted
                - Try with smaller files or simpler queries
                - Check internet connection for API calls
                """)

# --------------------------------------------------
# 9)  Footer Information
# --------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 System Status")
    st.success("✅ RAG Components Loaded")
    st.success("✅ Perplexity API Configured")
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Upload** your documents (PDF, DOCX, TXT, EML)
        2. **Adjust** retrieval settings in the sidebar
        3. **Enter** your question in the text box
        4. **Click** 'Run Query' to get AI-powered answers
        
        **Tips:**
        - Use specific questions for better results
        - Adjust similarity threshold if no results found
        - Try different keywords if the first attempt doesn't work
        """)
