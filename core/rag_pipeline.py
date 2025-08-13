
import streamlit as st
from typing import List, Dict
from .pdf_processor import PDFProcessor
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from models.groq_client import GroqClient
from config.settings import TOP_K_RETRIEVAL

class RAGPipeline:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.groq_client = GroqClient()
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load the FAISS index"""
        # Try to load existing index
        if not self.vector_store.load_index():
            # Create new index from book PDF
            self._create_index_from_book()
    
    def _create_index_from_book(self):
        """Create FAISS index from the book PDF"""
        with st.spinner("📚 Processing book PDF and creating search index..."):
            # Extract text from book PDF
            book_text = self.pdf_processor.extract_text_from_book()
            
            if not book_text:
                st.error("Could not extract text from book PDF")
                return
            
            # Chunk the text
            chunks = self.pdf_processor.chunk_text(book_text)
            
            if not chunks:
                st.error("Could not create chunks from book text")
                return
            
            # Generate embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            
            if len(embeddings) == 0:
                st.error("Could not generate embeddings")
                return
            
            # Create FAISS index
            self.vector_store.create_index(embeddings, chunks)
    
    def process_query(self, query: str) -> str:
        """Process user query and return response"""
        if not query.strip():
            return "Please ask a question about the book."
        
        try:
            with st.spinner("🔍 Searching for relevant information..."):
                # Generate query embedding
                query_embedding = self.embedding_generator.generate_single_embedding(query)
                
                if len(query_embedding) == 0:
                    return "Sorry, I couldn't process your query."
                
                # Search for relevant chunks
                relevant_chunks = self.vector_store.search(query_embedding, k=TOP_K_RETRIEVAL)
                
                if not relevant_chunks:
                    return "I couldn't find relevant information in the book to answer your question."
            
            with st.spinner("🤖 Generating response..."):
                # Generate response using Groq
                response = self.groq_client.generate_response(query, relevant_chunks)
                
                return response
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return "Sorry, I encountered an error while processing your question."
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index"""
        if self.vector_store.index is None:
            return {"status": "No index loaded", "documents": 0}
        
        return {
            "status": "Index loaded",
            "documents": len(self.vector_store.documents),
            "dimension": self.vector_store.index.d if self.vector_store.index else 0
        }
