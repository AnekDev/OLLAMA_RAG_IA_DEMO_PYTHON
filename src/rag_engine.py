import ollama
from chromadb import Client
from chromadb.config import Settings
import os


def safe_read_file(filepath):
    """Read a text file using multiple encodings until one works."""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for enc in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # Último recurso: reemplazar caracteres corruptos
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


class RAGEngine:
    def __init__(self, model_name, documents_path):
        self.model_name = model_name
        self.documents_path = documents_path
        
        # Inicializar ChromaDB
        self.client = Client(Settings())
        self.collection = self.client.create_collection("documents")


    def load_documents(self):
        """Load documents from the documents folder"""

        if not os.path.exists(self.documents_path):
            os.makedirs(self.documents_path)
            print(f"Documents folder created: {self.documents_path}")
            return
        
        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'):
                
                filepath = os.path.join(self.documents_path, filename)
                content = safe_read_file(filepath)   # ← AHORA content está definido
                
                # Embedding del documento
                embedding = ollama.embeddings(
                    model=self.model_name,
                    prompt=content
                )
                
                # Guardar en Chroma
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding["embedding"]],
                    ids=[filename]
                )
                
                print(f"Loaded: {filename}")

        print("All documents loaded.")


    def query(self, question):
        """Query the RAG system"""

        # Embedding de la pregunta
        question_embedding = ollama.embeddings(
            model=self.model_name,
            prompt=question
        )
        
        # Buscar documentos relevantes
        results = self.collection.query(
            query_embeddings=[question_embedding["embedding"]],
            n_results=3
        )
        
        # Contexto
        context = "\n\n".join(results["documents"][0])
        
        # Crear prompt final
        prompt = f"""
Context:
{context}

Question:
{question}

Answer based on the context above:
"""
        
        # Generar respuesta
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt
        )
        
        return response["response"]
