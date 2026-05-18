# RAG Chatbot - Copilot Instructions

This file contains workspace-specific instructions for GitHub Copilot.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) Chatbot application built with:
- **Backend**: FastAPI + LangChain + FAISS + OpenAI
- **Frontend**: Streamlit web interface
- **Purpose**: Answer questions about PDF documents using AI

## Project Structure

- `/backend/` - FastAPI application with RAG pipeline
- `/frontend/` - Streamlit UI
- `/data/` - PDF documents (user-provided)
- `/faiss_db/` - Vector store index (auto-generated)

## Setup Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Add OpenAI API key to `.env` file
3. Place PDF files in the `data/` directory
4. Run: `python app.py`

## Development Guidelines

### Backend Development
- Extend `/backend/main.py` for new API endpoints
- Modify `/backend/rag_pipeline.py` for RAG logic changes
- Update `/backend/vector_store.py` for vector store configuration

### Frontend Development
- Modify `/frontend/streamlit_app.py` for UI changes
- Add new tabs or interactive elements as needed

### Code Style
- Use type hints throughout
- Add docstrings to functions and classes
- Follow PEP 8 style guide

## Key Configuration

- **Vector Store Path**: `./faiss_db`
- **Data Directory**: `./data`
- **Backend Port**: `8000`
- **Frontend Port**: `8501`

## Common Tasks

### Add New API Endpoint
1. Edit `backend/main.py`
2. Add endpoint with appropriate Pydantic models
3. Test via Swagger UI at `http://localhost:8000/docs`

### Customize RAG Behavior
1. Edit `backend/rag_pipeline.py`
2. Modify the prompt template or retrieval settings
3. Reload documents to apply changes

### Enhance Frontend
1. Edit `frontend/streamlit_app.py`
2. Add new components or reorganize existing ones
3. Use Streamlit documentation for reference

## Testing

- API endpoints: Use Swagger UI or cURL
- Frontend: Interactive testing in Streamlit
- Vector store: Check `faiss_db/` directory

## Deployment Considerations

- Set `OPENAI_API_KEY` securely in production
- Use production-grade WSGI server instead of `--reload`
- Implement proper error handling and logging
- Add authentication if needed
- Use environment-specific configurations
