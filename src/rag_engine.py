import ollama
from chromadb import Client
from chromadb.config import Settings
import os

class RAGEngine:
    def __init__(self, model_name, docuemnts_path):
        self.model_name = model_name
        self.documents_path = docuemnts_path
        
        # Inicializar chromaDB
        self.client = Client(Settings())
        self.collection = self.client.create_collection("documents")
        
    def load_documents(self):
        """Load docuemnts from the documents folder"""
        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'): # Detectar los archivos .txt y leerlos
                with open(os.path.join(self.documents_path, filename), 'r') as f:
                    content = f.read()
                
                # Generate embedding with Ollama
                embedding = ollama.embeddings(
                    model=self.model_name,
                    prompt=content
                )
                
                # Store in Chroma
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding['embedding']],
                    ids=[filename]
                )
                
        print(f"Loaded {len(os.listdir(self.documents_path))} documents")
                
    def query(self, question):
        """Query the RAG system"""
        # Get question embedding
        question_embedding = ollama.embeddings(
            model = self.model_name,
            prompt = question
        )
                
        # Search for relevant documents
        results = self.collection.query(
            query_embeddings=[question_embedding['embedding']],
            n_results=3
        )
                
        # Build context from results
        context = "\n\n".join(results['documents'][0])
                
        # Generate answer with Ollama
        prompt = f"""Context: {context}
        
Question: {question}

Answer based on the context above"""

        response = ollama.generate(
            model = self.model_name,
            prompt = prompt
        )
        
        return response['response']