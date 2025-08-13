
import streamlit as st
from groq import Groq
from typing import List, Dict
from config.settings import GROQ_API_KEY, LLM_MODEL

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using Groq API with Mistral-7B"""
        try:
            # Prepare context from retrieved chunks
            context_text = self._prepare_context(context_chunks)
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided book content. 
            Use the context provided to answer the user's question. If the context doesn't contain enough information 
            to answer the question completely, mention that and provide what information you can from the context.
            Always be accurate and cite the page numbers when possible."""
            
            # Create user prompt with context
            user_prompt = f"""
            Context from the book:
            {context_text}
            
            Question: {query}
            
            Please provide a detailed answer based on the context above.
            """
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating the response."
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context text from retrieved chunks"""
        if not chunks:
            return "No relevant context found in the book."
        
        context_parts = []
        for chunk in chunks:
            page_info = f"[Page {chunk.get('page', 'Unknown')}]"
            chunk_text = chunk.get('text', '').strip()
            similarity = chunk.get('similarity_score', 0)
            
            context_parts.append(f"{page_info} (Relevance: {similarity:.3f})\n{chunk_text}")
        
        return "\n\n---\n\n".join(context_parts)
