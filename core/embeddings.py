
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from config.settings import EMBEDDING_MODEL, HUGGINGFACE_TOKEN

class EmbeddingGenerator:
    def __init__(self):
        self.model = None
        self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """Load the sentence transformer model"""
        try:
            model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HUGGINGFACE_TOKEN)
            return model
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.model = self._load_model()
        
        if self.model is None:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return np.array(embeddings)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return np.array([])
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0] if text else np.array([])
