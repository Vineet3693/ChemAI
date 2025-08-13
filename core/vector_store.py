
import faiss
import numpy as np
import pickle
import os
import streamlit as st
from typing import List, Dict
from config.settings import FAISS_INDEX_PATH

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.index_path = os.path.join(FAISS_INDEX_PATH, "faiss.index")
        self.docs_path = os.path.join(FAISS_INDEX_PATH, "documents.pkl")
    
    def create_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """Create FAISS index from embeddings and documents"""
        if len(embeddings) == 0:
            st.error("No embeddings provided for indexing")
            return
        
        try:
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store documents
            self.documents = documents
            
            # Save to disk
            self.save_index()
            
            st.success(f"Created FAISS index with {len(documents)} documents")
            
        except Exception as e:
            st.error(f"Error creating FAISS index: {str(e)}")
    
    def load_index(self) -> bool:
        """Load existing FAISS index from disk"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                return True
        except Exception as e:
            st.warning(f"Could not load existing index: {str(e)}")
        return False
    
    def save_index(self):
        """Save FAISS index to disk"""
        try:
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            st.error(f"Error saving index: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    result = self.documents[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Error searching index: {str(e)}")
            return []
