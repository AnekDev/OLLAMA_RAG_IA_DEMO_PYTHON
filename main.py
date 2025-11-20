from src.rag_engine import RAGEngine

folderDocumentsPath = "data/documents"

def main():
    #Inizializar 
    rag = RAGEngine(
        model_name="llama2",
        documents_path=folderDocumentsPath
    )

    # Cargar y processar los documentos
    rag.load_documents()
    
    while True:
        query = input("\nPreguntame algo (o escribe 'quit' para salir)")
        if query.lower() == 'quit':
            break
        
        response = rag.query(query)
        print(f"Respuesta: {response}")
        
if __name__ == "__main__":
    main()