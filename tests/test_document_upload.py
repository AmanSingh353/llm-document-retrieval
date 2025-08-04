# tests/test_document_upload.py
import os
from src.data.loader import load_data, load_single_file

def test_document_loading():
    """Test document loading with different file types"""
    
    # Create test files directory
    test_files_dir = "test_documents"
    
    if not os.path.exists(test_files_dir):
        print(f"Create a '{test_files_dir}' directory with sample files:")
        print("- sample.pdf")
        print("- sample.docx") 
        print("- sample.txt")
        print("- sample.eml")
        return
    
    print("🧪 Testing Document Loading...")
    
    # Test loading entire directory
    docs = load_data(test_files_dir)
    print(f"✅ Loaded {len(docs)} documents from directory")
    
    # Test individual file types
    for filename in os.listdir(test_files_dir):
        filepath = os.path.join(test_files_dir, filename)
        try:
            docs = load_single_file(filepath)
            print(f"✅ {filename}: {len(docs)} chunks loaded")
        except Exception as e:
            print(f"❌ {filename}: Error - {str(e)}")

if __name__ == "__main__":
    test_document_loading()
