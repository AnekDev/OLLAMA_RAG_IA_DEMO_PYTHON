import ollama
from typing import List

class EmbeddingGenerator:
    """Generate embeddings using Ollama"""
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for i, text in enumerate(texts):
            print(f"Generating embedding {i+1}/{len(texts)}...")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings