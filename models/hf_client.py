
import streamlit as st
from huggingface_hub import InferenceClient
from config.settings import HUGGINGFACE_TOKEN

class HuggingFaceClient:
    def __init__(self):
        self.token = HUGGINGFACE_TOKEN
        self.client = InferenceClient(token=self.token)
    
    def test_connection(self) -> bool:
        """Test HuggingFace API connection"""
        try:
            # Test with a simple embedding request
            response = self.client.feature_extraction(
                "test connection",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            return True
        except Exception as e:
            st.error(f"HuggingFace connection test failed: {str(e)}")
            return False
