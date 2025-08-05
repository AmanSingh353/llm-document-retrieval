import os
import hashlib
import sqlite3
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DB_FILE = 'db/metadata.db'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('db', exist_ok=True)

# Initialize DB
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS file_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            user TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sha256 TEXT NOT NULL
        );
        ''')
        conn.commit()

init_db()

# Helper to compute SHA-256
def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    user = request.form.get('user')

    if not file or not user:
        return jsonify({'error': 'Missing file or user'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    file_hash = compute_sha256(filepath)

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            INSERT INTO file_metadata (filename, user, sha256)
            VALUES (?, ?, ?)
        ''', (file.filename, user, file_hash))
        conn.commit()

    return jsonify({'message': 'File uploaded', 'sha256': file_hash}), 200

if __name__ == '__main__':
    app.run(debug=True)
