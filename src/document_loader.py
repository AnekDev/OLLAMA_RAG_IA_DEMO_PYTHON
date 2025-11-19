import os
from typing import List, Dict

class DocumentLoader:
    """Load and process documents from a directory"""
    
    def __init__(self, documents_path: str):
        self.documents_path = documents_path
        
    def load_txt_files(self) -> List[Dict[str, str]]:
        """Load all .txt files from the documents directory"""
        documents = []
        
        if not os.path.exists(self.documents_path):
            os.makedirs(self.documents_path)
            print(f"Created documents directory at {self.documents_path}")
            return documents
        
        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.documents_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'id': filename,
                            'content': content,
                            'source': filepath
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks