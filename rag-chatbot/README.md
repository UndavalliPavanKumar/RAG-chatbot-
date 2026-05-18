# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot application that answers questions about PDF documents using LangChain, FAISS, and OpenAI's GPT models.

## Features

- **PDF Processing**: Automatically loads and processes PDF documents from the `data/` directory
- **Vector Store**: Uses FAISS for efficient similarity search over document embeddings
- **RAG Pipeline**: Retrieves relevant documents and generates contextual answers using GPT
- **FastAPI Backend**: RESTful API with document management and querying endpoints
- **Streamlit Frontend**: User-friendly web interface for interactive chatting
- **Document Management**: Reload documents on-demand without restarting the application

## Project Structure

```
rag-chatbot/
├── app.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── backend/
│   ├── main.py              # FastAPI application
│   ├── rag_pipeline.py      # RAG implementation
│   ├── vector_store.py      # FAISS vector store management
│   └── pdf_loader.py        # PDF document loading
├── data/                    # PDF documents (add your PDFs here)
├── faiss_db/               # Vector store index (auto-generated)
└── frontend/
    └── streamlit_app.py    # Streamlit web interface
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd rag-chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Add your PDF documents**
   - Place PDF files in the `data/` directory

## Usage

### Option 1: Run Both Backend and Frontend

```bash
python app.py
```

This starts:
- FastAPI backend on `http://localhost:8000`
- Streamlit frontend on `http://localhost:8501`

### Option 2: Run Services Separately

**Backend:**
```bash
python -m uvicorn backend.main:app --reload
```
API documentation available at `http://localhost:8000/docs`

**Frontend:**
```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

### Core Endpoints

- **POST `/query`** - Ask a question
  ```json
  {
    "question": "What is this document about?"
  }
  ```
  
- **POST `/reload-documents`** - Reload documents from data directory

- **GET `/health`** - Health check status

- **GET `/`** - API information

### Interactive API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Edit `.env` to customize:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_key_here

# Vector Store Path
VECTOR_STORE_PATH=./faiss_db
DATA_DIR=./data

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## Dependencies

- **langchain** - RAG and LLM orchestration
- **faiss-cpu** - Vector similarity search
- **openai** - GPT models and embeddings
- **fastapi** - REST API framework
- **streamlit** - Web UI framework
- **pypdf** - PDF parsing
- **python-dotenv** - Environment variable management

## Workflow

1. **Document Loading**: PDF files are loaded from the `data/` directory
2. **Embedding**: Documents are split into chunks and embedded using OpenAI's embedding model
3. **Vector Store**: Embeddings are stored in FAISS for efficient retrieval
4. **Query Processing**:
   - User submits a question
   - Similar documents are retrieved using vector similarity search
   - Retrieved documents are passed to GPT with the question
   - GPT generates an answer based on the context

## Troubleshooting

### Connection Errors
- Ensure the backend is running on `localhost:8000`
- Check firewall settings

### No Documents Found
- Verify PDF files are in the `data/` directory
- Use the "Reload Documents" button in the UI

### OpenAI API Errors
- Verify your API key is correct in `.env`
- Check your OpenAI account has available credits
- Ensure you have proper API access permissions

### Vector Store Issues
- Delete the `faiss_db/` folder to reset the index
- Use the "Reload Documents" button to rebuild it

## Development

### Adding Custom Prompts

Edit the prompt template in `backend/rag_pipeline.py`:

```python
prompt_template = """Your custom prompt here..."""
```

### Extending the API

Add new endpoints in `backend/main.py` following FastAPI patterns.

### Customizing the UI

Modify `frontend/streamlit_app.py` to change the interface.

## Performance Tips

- Use smaller PDF files for faster processing
- Adjust the number of retrieved documents in `vector_store.search(k=4)`
- Consider using `gpt-3.5-turbo` for faster responses
- Batch process large document collections

## License

This project is provided as-is for educational and development purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review FastAPI documentation: https://fastapi.tiangolo.com/
3. Review Streamlit documentation: https://docs.streamlit.io/
4. Review LangChain documentation: https://python.langchain.com/

## Future Enhancements

- [ ] Multi-file upload UI
- [ ] Document preview and management
- [ ] Chat history and context management
- [ ] Custom model selection
- [ ] Rate limiting and user authentication
- [ ] Database integration for persistence
- [ ] Advanced filtering and search
- [ ] Batch document processing
