import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables reliably
load_dotenv(find_dotenv())

# Project settings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'documents')

# Ollama settings
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')

# RAG settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 3))
