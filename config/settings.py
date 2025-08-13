
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "xyz")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "rst")

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mixtral-8x7b-32768"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# Paths
DATA_PATH = "data"
PDF_PATH = os.path.join(DATA_PATH, "book.pdf")
FAISS_INDEX_PATH = os.path.join(DATA_PATH, "faiss_indexes")
CACHE_PATH = os.path.join(DATA_PATH, "cache")

# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)
