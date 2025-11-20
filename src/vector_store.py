import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
    """Manage vector storage and retrieval with ChromaDB"""
    
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client(Settings())
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(collection_name)
        
    def add_documents(self, documents: List[str], embeddings: List[List[float]], ids: List[str]):
        """Add documents with their embeddings to the store"""
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            print(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def search(self, query_embedding: List[float], n_results: int = 3) -> Dict:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return {'documents': [[]], 'distances': [[]], 'ids': [[]]}
    
    def get_count(self) -> int:
        """Get number of documents in the store"""
        return self.collection.count()