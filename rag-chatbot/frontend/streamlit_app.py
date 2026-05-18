"""Streamlit frontend for the RAG Chatbot."""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"


def query_backend(question: str) -> dict:
    """Send a query to the backend API.
    
    Args:
        question: The question to ask
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Error: Cannot connect to backend API. Make sure the backend is running on localhost:8000",
            "sources": []
        }
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }


def health_check() -> bool:
    """Check if the backend API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main Streamlit application."""
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about your documents")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Health check
        col1, col2 = st.columns(2)
        with col1:
            if health_check():
                st.success("✅ Backend Connected")
            else:
                st.error("❌ Backend Disconnected")
        
        with col2:
            if st.button("🔄 Reload Documents"):
                try:
                    response = requests.post(f"{API_URL}/reload-documents", timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data.get('message', 'Documents reloaded')}")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error reloading documents: {str(e)}")
        
        st.divider()
        st.markdown("### About")
        st.info("This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions based on your documents.")
    
    # Main content
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📚 Document Search"])
    
    with tab1:
        # Chat interface
        st.markdown("### Ask a Question")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="What is this document about?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Ask Question", type="primary", use_container_width=True):
                if question.strip():
                    with st.spinner("Searching and generating answer..."):
                        result = query_backend(question)
                        
                        st.markdown("### Answer")
                        st.write(result["answer"])
                        
                        if result.get("sources"):
                            st.markdown("### Sources")
                            for source in result["sources"]:
                                st.caption(f"📄 {source}")
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
    
    with tab2:
        st.markdown("### Search Documents")
        
        search_query = st.text_input(
            "Search for relevant documents:",
            placeholder="Enter search terms"
        )
        
        num_results = st.slider("Number of results:", min_value=1, max_value=10, value=4)
        
        if st.button("🔎 Search", type="primary", use_container_width=True):
            if search_query.strip():
                try:
                    # This would require an additional endpoint in the backend
                    st.info("Document search functionality can be implemented by adding a search endpoint to the backend.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a search query")
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.85em;'>RAG Chatbot v1.0 | Powered by LangChain, FAISS & OpenAI</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
