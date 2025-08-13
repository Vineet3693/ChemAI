
import streamlit as st
import os
from utils.session_state import initialize_session_state, get_rag_pipeline, add_message, clear_chat_history
from utils.helpers import check_book_pdf_exists, get_pdf_info, validate_api_keys, create_data_directories
from config.settings import PDF_PATH

# Page configuration
st.set_page_config(
    page_title="📚 RAG Book Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Create necessary directories
    create_data_directories()
    
    # Header
    st.title("📚 RAG Book Chatbot")
    st.markdown("Ask questions about your book and get AI-powered answers!")
    
    # Sidebar
    render_sidebar()
    
    # Main chat interface
    render_main_interface()

def render_sidebar():
    """Render sidebar with book info and settings"""
    with st.sidebar:
        st.header("📖 Book Information")
        
        # Book PDF status
        pdf_info = get_pdf_info()
        if pdf_info['exists']:
            st.success(f"✅ Book loaded: {pdf_info['name']}")
            st.info(f"📄 Size: {pdf_info['size_mb']} MB")
        else:
            st.error("❌ No book PDF found!")
            st.info(f"Please add your book PDF as 'book.pdf' in the '{os.path.dirname(PDF_PATH)}' folder")
        
        st.divider()
        
        # API Status
        st.header("🔑 API Status")
        api_status = validate_api_keys()
        
        if api_status['groq']:
            st.success("✅ Groq API Key")
        else:
            st.error("❌ Groq API Key missing")
        
        if api_status['huggingface']:
            st.success("✅ HuggingFace Token")
        else:
            st.error("❌ HuggingFace Token missing")
        
        st.divider()
        
        # Index Status
        st.header("📊 Index Status")
        if st.session_state.get('index_ready', False):
            rag = get_rag_pipeline()
            stats = rag.get_index_stats()
            st.success(f"✅ {stats['status']}")
            st.info(f"📄 Documents: {stats['documents']}")
        else:
            st.warning("⏳ Index not ready")
        
        st.divider()
        
        # Controls
        st.header("🎛️ Controls")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
        
        if st.button("🔄 Rebuild Index", use_container_width=True):
            if check_book_pdf_exists():
                st.session_state.rag_pipeline = None
                st.session_state.index_ready = False
                st.rerun()
            else:
                st.error("Cannot rebuild index: No book PDF found!")

def render_main_interface():
    """Render main chat interface"""
    
    # Check prerequisites
    if not check_book_pdf_exists():
        st.error("❌ Please add your book PDF to the data folder first!")
        st.stop()
    
    api_status = validate_api_keys()
    if not all(api_status.values()):
        st.error("❌ Please set your API keys in the .env file!")
        st.stop()
    
    # Initialize RAG pipeline
    if not st.session_state.get('index_ready', False):
        with st.spinner("🔄 Initializing RAG system..."):
            try:
                rag = get_rag_pipeline()
                st.success("✅ RAG system ready!")
            except Exception as e:
                st.error(f"❌ Error initializing RAG system: {str(e)}")
                st.stop()
    
    # Chat interface
    st.subheader("💬 Chat with your Book")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**{i}. Page {source.get('page', 'Unknown')}** "
                                f"(Relevance: {source.get('similarity_score', 0):.3f})")
                        st.write(f"_{source.get('text', '')[:200]}..._")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the book..."):
        # Add user message
        add_message("user", prompt)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                rag = get_rag_pipeline()
                response = rag.process_query(prompt)
                st.write(response)
                
                # Add assistant message to history
                add_message("assistant", response)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                add_message("assistant", error_msg)
        
        # Rerun to update the interface
        st.rerun()

if __name__ == "__main__":
    main()
