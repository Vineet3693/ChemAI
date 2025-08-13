
import PyPDF2
import streamlit as st
from typing import List
import os
from config.settings import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
    
    def extract_text_from_book(self) -> str:
        """Extract text from the book PDF in data folder"""
        if not os.path.exists(PDF_PATH):
            st.error(f"Book PDF not found at {PDF_PATH}")
            return ""
        
        try:
            with open(PDF_PATH, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[dict]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Find which page this chunk belongs to
            page_num = self._find_page_number(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "chunk_id": len(chunks)
            })
        
        return chunks
    
    def _find_page_number(self, chunk_text: str) -> int:
        """Extract page number from chunk text"""
        import re
        page_match = re.search(r"--- Page (\d+) ---", chunk_text)
        return int(page_match.group(1)) if page_match else 1
