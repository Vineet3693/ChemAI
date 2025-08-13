
import streamlit as st
from typing import List, Dict
import os
from config.settings import DATA_PATH, PDF_PATH

def check_book_pdf_exists() -> bool:
    """Check if the book PDF exists in data folder"""
    return os.path.exists(PDF_PATH)

def get_pdf_info() -> Dict:
    """Get information about the book PDF"""
    if not check_book_pdf_exists():
        return {"exists": False, "size": 0, "name": ""}
    
    try:
        size = os.path.getsize(PDF_PATH)
        name = os.path.basename(PDF_PATH)
        return {
            "exists": True,
            "size": size,
            "name": name,
            "size_mb": round(size / (1024 * 1024), 2)
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}

def format_sources(sources: List[Dict]) -> str:
    """Format source information for display"""
    if not sources:
        return "No sources found"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        page = source.get('page', 'Unknown')
        score = source.get('similarity_score', 0)
        formatted.append(f"{i}. Page {page} (Relevance: {score:.3f})")
    
    return "\n".join(formatted)

def validate_api_keys() -> Dict[str, bool]:
    """Validate that API keys are set"""
    from config.settings import GROQ_API_KEY, HUGGINGFACE_TOKEN
    
    return {
        'groq': bool(GROQ_API_KEY and GROQ_API_KEY != 'xyz'),
        'huggingface': bool(HUGGINGFACE_TOKEN and HUGGINGFACE_TOKEN != 'rst')
    }

def create_data_directories():
    """Ensure all necessary data directories exist"""
    directories = [
        DATA_PATH,
        os.path.join(DATA_PATH, 'faiss_indexes'),
        os.path.join(DATA_PATH, 'cache')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
