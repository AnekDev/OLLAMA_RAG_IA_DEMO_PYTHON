from src.rag_engine import RAGEngine

folderDocumentsPath = "data/docuemnts" # La ruta a la carpeta donde se va a usar

def main():
    #Inizializar 
    rag = RAGEngine (
        model_name="llama2", # Aquí se decide el modelo de IA que vas a hacer, esto se deberia poder cambiar más adelante dependiendo del servicio que quieras
        documents_path = folderDocumentsPath # Definimos de que carpeta va a sacar la informacion la IA 
    )
    
    # Cargar y processar los documentos
    rag.load_documents()
    
    while True:
        query = input("\nPreguntame algo (o escribe 'quit' para salir)")
        if query.lower() == 'quit':
            break
        
        response = rag.query(query)
        print(f"\Respuesta: {response}")
        
if __name__ == "__main__":
    main()