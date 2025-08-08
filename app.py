import os
import sys
import hashlib
import sqlite3
import tempfile
import io
from datetime import datetime

# Add the project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Flask and other standard imports
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Try to import RAG components with error handling
try:
    from src.data.loader import load_data
    from src.retrieval.retriever import Retriever
    from src.llm.llm_interface import LLMInterface
    from src.types.index import QueryInput
    
    RAG_AVAILABLE = True
    print("✅ RAG components loaded successfully")
except ImportError as e:
    print(f"❌ Could not import RAG components: {e}")
    print("⚠️  RAG functionality will be disabled")
    RAG_AVAILABLE = False

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DB_FILE = 'db/metadata.db'
API_KEY = os.getenv("API_KEY", "your_api_key_here")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('db', exist_ok=True)

if not PERPLEXITY_API_KEY and RAG_AVAILABLE:
    print("WARNING: PERPLEXITY_API_KEY not found in environment variables")

# Initialize DB with enhanced schema
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS file_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            user TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sha256 TEXT NOT NULL,
            file_size INTEGER,
            content_type TEXT,
            processed BOOLEAN DEFAULT FALSE,
            processing_error TEXT
        );
        ''')
        
        # Add indexes for better performance
        conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_sha256 ON file_metadata(sha256);
        ''')
        conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_user ON file_metadata(user);
        ''')
        conn.commit()

init_db()

# Helper functions
def compute_sha256(file_path):
    """Compute SHA-256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def authenticate_request():
    """Check API key authentication"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    token = auth_header.split(' ')[1]
    return token == API_KEY

# Routes

@app.route('/', methods=['GET'])
def index():
    """API information endpoint"""
    return jsonify({
        "message": "LLM Document Retrieval System",
        "version": "1.0.0",
        "rag_available": RAG_AVAILABLE,
        "endpoints": {
            "/upload": "POST - Upload documents",
            "/query": "POST - Query documents (requires RAG)",
            "/api/v1/hackrx/run": "POST - Process PDF from URL and answer questions (requires RAG)",
            "/files": "GET - List uploaded files",
            "/health": "GET - Health check"
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint (existing functionality)"""
    file = request.files.get('file')
    user = request.form.get('user')

    if not file or not user:
        return jsonify({'error': 'Missing file or user'}), 400

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Compute metadata
    file_hash = compute_sha256(filepath)
    file_size = os.path.getsize(filepath)
    content_type = file.content_type or 'application/octet-stream'

    # Store in database
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            INSERT INTO file_metadata (filename, user, sha256, file_size, content_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename, user, file_hash, file_size, content_type))
        conn.commit()

    return jsonify({
        'message': 'File uploaded successfully',
        'filename': file.filename,
        'sha256': file_hash,
        'size': file_size,
        'user': user
    }), 200

@app.route('/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    user_filter = request.args.get('user')
    
    query = 'SELECT id, filename, user, upload_time, sha256, file_size, processed FROM file_metadata'
    params = []
    
    if user_filter:
        query += ' WHERE user = ?'
        params.append(user_filter)
    
    query += ' ORDER BY upload_time DESC'
    
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        files = [dict(row) for row in cursor.fetchall()]
    
    return jsonify({'files': files})

@app.route('/query', methods=['POST'])
def query_documents():
    """Query uploaded documents using RAG system"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not RAG_AVAILABLE:
        return jsonify({'error': 'RAG system not available - missing dependencies'}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    query_text = data.get('query')
    k = data.get('k', 5)
    similarity_threshold = data.get('similarity_threshold', 0.4)
    filters = data.get('filters', {})
    
    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400
    
    try:
        if not PERPLEXITY_API_KEY:
            return jsonify({'error': 'LLM service not configured'}), 500
        
        # Load documents from uploads folder
        documents = load_data(UPLOAD_FOLDER)
        
        if not documents:
            return jsonify({
                'answer': 'No documents found in the system',
                'relevant_chunks': [],
                'justification': 'No documents available for querying'
            })
        
        # Initialize RAG system
        retriever = Retriever(documents)
        llm = LLMInterface(perplexity_api_key=PERPLEXITY_API_KEY)
        
        # Process query
        parsed_query = QueryInput(raw_query=query_text)
        relevant_chunks = retriever.retrieve(
            parsed_query.raw_query,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        if not relevant_chunks:
            return jsonify({
                'answer': 'No relevant information found for your query',
                'relevant_chunks': [],
                'justification': 'No matching content retrieved'
            })
        
        # Get LLM response
        llm_response = llm.process_query(parsed_query, relevant_chunks)
        
        # Extract chunk texts
        chunk_texts = []
        for chunk in relevant_chunks:
            if hasattr(chunk, 'content'):
                chunk_texts.append(chunk.content)
            elif isinstance(chunk, dict):
                chunk_texts.append(chunk.get('content', str(chunk)))
            else:
                chunk_texts.append(str(chunk))
        
        # Format response
        if isinstance(llm_response, dict):
            response = {
                'answer': llm_response.get('answer', 'No answer generated'),
                'justification': llm_response.get('justification', f'Based on {len(relevant_chunks)} relevant sections'),
                'relevant_chunks': chunk_texts[:3],
                'total_chunks_found': len(relevant_chunks),
                'parameters': {
                    'k': k,
                    'similarity_threshold': similarity_threshold
                }
            }
        else:
            response = {
                'answer': str(llm_response),
                'relevant_chunks': chunk_texts[:3],
                'justification': f'Based on {len(relevant_chunks)} relevant sections'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/v1/hackrx/run', methods=['POST'])
def hackrx_endpoint():
    """FastAPI-compatible endpoint for PDF URL processing"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not RAG_AVAILABLE:
        return jsonify({'error': 'RAG system not available - missing dependencies'}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    documents_url = data.get('documents')
    questions = data.get('questions', [])
    k = data.get('k', 5)
    similarity_threshold = data.get('similarity_threshold', 0.4)
    
    if not documents_url or not questions:
        return jsonify({'error': 'Missing documents URL or questions'}), 400
    
    try:
        # Download PDF
        print(f"Downloading PDF from: {documents_url}")
        response = requests.get(documents_url, timeout=30)
        if response.status_code != 200:
            return jsonify({'error': f'Could not download PDF. Status: {response.status_code}'}), 400
        
        # Process PDF using existing pipeline
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = os.path.join(tmp_dir, "document.pdf")
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            
            print("Processing PDF...")
            documents = load_data(tmp_dir)
            if not documents:
                return jsonify({'error': 'No content extracted from PDF'}), 400
            
            retriever = Retriever(documents)
            llm = LLMInterface(perplexity_api_key=PERPLEXITY_API_KEY)
            
            answers = []
            for question in questions:
                print(f"Processing question: {question}")
                try:
                    parsed_query = QueryInput(raw_query=question)
                    relevant_chunks = retriever.retrieve(
                        parsed_query.raw_query, k=k, similarity_threshold=similarity_threshold
                    )
                    
                    if not relevant_chunks:
                        answers.append({
                            "question": question,
                            "answer": "No relevant information found",
                            "relevant_chunks": [],
                            "confidence": 0.0
                        })
                        continue
                    
                    llm_response = llm.process_query(parsed_query, relevant_chunks)
                    
                    chunk_texts = []
                    for chunk in relevant_chunks:
                        if hasattr(chunk, 'content'):
                            chunk_texts.append(chunk.content)
                        elif isinstance(chunk, dict):
                            chunk_texts.append(chunk.get('content', str(chunk)))
                        else:
                            chunk_texts.append(str(chunk))
                    
                    confidence = min(len(relevant_chunks) / k, 1.0)
                    
                    if isinstance(llm_response, dict):
                        answer_text = llm_response.get("answer", "No answer generated")
                    else:
                        answer_text = str(llm_response)
                    
                    answers.append({
                        "question": question,
                        "answer": answer_text,
                        "relevant_chunks": chunk_texts[:3],
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    print(f"Error processing question '{question}': {str(e)}")
                    answers.append({
                        "question": question,
                        "answer": f"Error: {str(e)}",
                        "relevant_chunks": [],
                        "confidence": 0.0
                    })
            
            return jsonify({
                "answers": answers,
                "document_processed": True,
                "total_questions": len(questions)
            })
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
        'database_exists': os.path.exists(DB_FILE),
        'rag_available': RAG_AVAILABLE,
        'perplexity_configured': bool(PERPLEXITY_API_KEY) if RAG_AVAILABLE else None
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Database: {DB_FILE}")
    print(f"RAG system available: {RAG_AVAILABLE}")
    if RAG_AVAILABLE and PERPLEXITY_API_KEY:
        print("✅ All systems ready")
    elif not RAG_AVAILABLE:
        print("⚠️  Running in basic mode (file upload only)")
    app.run(host='127.0.0.1', port=5000, debug=True)
