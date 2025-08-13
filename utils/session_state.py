
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    
    # Index status
    if 'index_ready' not in st.session_state:
        st.session_state.index_ready = False
    
    # API status
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {
            'groq': False,
            'huggingface': False
        }

def get_chat_history() -> List[Dict[str, Any]]:
    """Get formatted chat history"""
    return st.session_state.messages

def add_message(role: str, content: str, sources: List[Dict] = None):
    """Add message to chat history"""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now(),
        'sources': sources or []
    }
    st.session_state.messages.append(message)

def clear_chat_history():
    """Clear all chat messages"""
    st.session_state.messages = []

def get_rag_pipeline():
    """Get or initialize RAG pipeline"""
    if st.session_state.rag_pipeline is None:
        from core.rag_pipeline import RAGPipeline
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.index_ready = True
    
    return st.session_state.rag_pipeline
