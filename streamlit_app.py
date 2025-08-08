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
# HELPER FUNCTIONS (MOVED TO TOP TO FIX ERRORS)
# --------------------------------------------------
def display_upload_results(response, relevant_chunks, query, k, similarity_threshold, uploaded_files, filters):
    """Display results for upload mode"""
    
    # Main response section
    st.subheader("✅ Query Results")
    
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

# --------------------------------------------------
# 5) Mode Selection
# --------------------------------------------------
tab1, tab2 = st.tabs(["📁 Upload Documents", "🌐 PDF URL Query"])

# --------------------------------------------------
# 6) Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Show different settings based on selected tab
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = 0
    
    # Common settings
    st.markdown("### 🔍 Retrieval Settings")
    
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
    
    # Advanced settings for upload mode
    if RAG_COMPONENTS_LOADED:
        st.markdown("### 🔍 Optional Filters")
        
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
        
        st.markdown("### 📊 Model Settings")
        
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
# TAB 1: Upload Documents Interface
# --------------------------------------------------
with tab1:
    if not RAG_COMPONENTS_LOADED:
        st.error("❌ RAG components not loaded. Please check your project structure.")
        st.stop()
    
    st.markdown("### 📁 Upload and Query Documents")
    
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g. What is the grace period for premium payments?",
        help="Ask specific questions about your uploaded documents",
        key="upload_query"
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

    # Processing Section for Upload Mode
    if st.button("🔍 Run Query", type="primary", key="upload_run"):
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
                    status_text.text("🤖 Generating Query Result...")
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

                # Display Results
                display_upload_results(response, relevant_chunks, query, k, similarity_threshold, uploaded_files, filters)

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
# TAB 2: PDF URL Interface
# --------------------------------------------------
with tab2:
    st.markdown("### 🌐 Query PDF from URL")
    
    pdf_url = st.text_input(
        "Enter the PDF URL:",
        placeholder="https://example.com/document.pdf",
        help="Provide a direct URL to a PDF document",
        key="pdf_url"
    )
    
    questions = st.text_area(
        "Enter your questions (one per line):",
        placeholder="What is the main topic?\nWhat are the key findings?\nSummarize the conclusions.",
        help="Enter multiple questions, each on a new line",
        key="pdf_questions"
    )
    
    # API Configuration section
    with st.expander("🔧 API Configuration"):
        custom_api_url = st.text_input("Custom API URL:", value=API_URL)
        custom_api_key = st.text_input("API Key:", value=API_KEY, type="password")
        timeout = st.number_input("Request timeout (seconds):", min_value=10, max_value=120, value=60)

    if st.button("🔍 Get Answers", type="primary", key="pdf_run"):
        if not pdf_url or not questions:
            st.warning("⚠️ Please enter both PDF URL and questions.")
        else:
            question_list = [q.strip() for q in questions.split("\n") if q.strip()]
            
            if not question_list:
                st.warning("⚠️ No valid questions found.")
                st.stop()
            
            payload = {
                "documents": pdf_url,
                "questions": question_list
            }
            
            # Show processing status
            with st.spinner(f"🔄 Processing {len(question_list)} questions from PDF..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📡 Sending request to API...")
                    progress_bar.progress(25)
                    
                    response = requests.post(
                        custom_api_url or API_URL,
                        headers={
                            "Authorization": f"Bearer {custom_api_key or API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=timeout
                    )
                    
                    progress_bar.progress(75)
                    status_text.text("🤖 Processing Query Result...")
                    
                    if response.status_code == 200:
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        display_pdf_results(response.json(), question_list, pdf_url)
                        
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        with st.expander("🐛 Error Details"):
                            st.code(response.text, language="json")
                            
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. The PDF might be too large or the server is busy.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection error. Please check your internet connection and API URL.")
                except Exception as e:
                    st.error(f"❌ Request failed: {str(e)}")
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc(), language="python")

# --------------------------------------------------
# Footer Information
# --------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 System Status")
    
    if RAG_COMPONENTS_LOADED:
        st.success("✅ RAG Components Loaded")
    else:
        st.error("❌ RAG Components Failed")
        
    st.success("✅ Perplexity API Configured")
    
    # API status check for PDF mode
    with st.expander("🌐 API Status"):
        st.write(f"**PDF API URL:** {API_URL}")
        st.write("**Status:** Ready for PDF processing")
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Upload Documents Mode:**
        1. Upload your documents (PDF, DOCX, TXT, EML)
        2. Adjust retrieval settings in the sidebar
        3. Enter your question in the text box
        4. Click 'Run Query' to get AI-powered answers
        
        **PDF URL Mode:**
        1. Enter a direct URL to a PDF document
        2. Type your questions (one per line)
        3. Configure API settings if needed
        4. Click 'Get Answers' for processing
        
        **Tips:**
        - Use specific questions for better results
        - Adjust similarity threshold if no results found
        - Try different keywords if the first attempt doesn't work
        """)
