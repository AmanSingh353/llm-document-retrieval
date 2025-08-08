import os, hashlib, sqlite3, uuid
from datetime import datetime
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from chromadb import PersistentClient
from dotenv import load_dotenv

from chromadb import PersistentClient

# ✅ Create or connect to a persistent vector store
client = PersistentClient(path="vector_store")  # This folder will store your vectors permanently

# ✅ Get or create your collection (like a table in DB)
collection = client.get_or_create_collection(name="documents")

# Setup
load_dotenv()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
executor = ThreadPoolExecutor()
client = PersistentClient(path="vector_store/")
collection = client.get_or_create_collection("documents")

DB_PATH = 'db/metadata.db'

# Utility: SHA-256
def compute_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# Background task
def process_file(file_path, filename, file_hash):
    # Avoid duplicate embeddings
    print(f"📄 Processing: {filename} (hash: {file_hash})")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        collection.add(
            documents=[text],
            metadatas=[{"filename": filename}],
            ids=[file_hash]
        )
        print(f"✅ Embedded: {filename}")
    except Exception as e:
        print(f"⚠️ Error during embedding: {e}")

@app.route("/upload", methods=["POST"])
def query_vectors():
    data = request.get_json()
    query_text = data.get("query", "")
    k = int(data.get("k", 5))
    threshold = float(data.get("similarity_threshold", 0.75))
    filters = data.get("filters", {})

    metadata_filter = {}

    if "filename" in filters:
        metadata_filter["filename"] = filters["filename"]
    if "user_role" in filters:
        metadata_filter["user_role"] = filters["user_role"]

    if "upload_date_from" in filters or "upload_date_to" in filters:
        from datetime import datetime
        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").isoformat()
            except:
                return None
        upload_from = parse_date(filters.get("upload_date_from", "1900-01-01"))
        upload_to = parse_date(filters.get("upload_date_to", "2100-01-01"))
        metadata_filter["upload_date"] = {
            "$gte": upload_from,
            "$lte": upload_to
        }

    # Connect to your Chroma vector store
    from chromadb import PersistentClient
    client = PersistentClient(path="vector_store")
    collection = client.get_or_create_collection("documents")

    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        where=metadata_filter
    )

    filtered_results = []
    for doc, score, metadata in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        if score >= threshold:
            filtered_results.append({
                "document": doc,
                "similarity": score,
                "metadata": metadata
            })

    return jsonify(filtered_results)

    # Check hash in DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM file_metadata WHERE hash = ?", (file_hash,))
    exists = c.fetchone()

    if exists:
        print(f"⚠️ Duplicate file: {filename}")
        os.remove(temp_path)
        conn.close()
        return jsonify({"status": "duplicate", "hash": file_hash}), 200

    # Insert metadata
    c.execute(
        "INSERT INTO file_metadata (filename, hash, upload_time) VALUES (?, ?, ?)",
        (filename, file_hash, datetime.now())
    )
    conn.commit()
    conn.close()

    # Persist and process in background
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    os.rename(temp_path, saved_path)
    executor.submit(process_file, saved_path, filename, file_hash)

    return jsonify({"status": "uploaded", "hash": file_hash}), 201

if __name__ == "__main__":
    app.run(debug=True)
