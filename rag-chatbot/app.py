"""Main entry point for the RAG Chatbot application."""

import os
import subprocess
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Get the correct Python executable from the virtual environment
def get_python_executable():
    """Get the Python executable path, preferring virtual environment if available."""
    venv_dir = Path(__file__).parent.parent / ".venv"
    if venv_dir.exists():
        python_exe = venv_dir / "Scripts" / "python.exe"
        if python_exe.exists():
            return str(python_exe)
    
    # Fallback to current executable
    return sys.executable


def run_backend():
    """Run the FastAPI backend server."""
    print("Starting backend server...")
    python_exe = get_python_executable()
    subprocess.Popen([
        python_exe, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


def run_frontend():
    """Run the Streamlit frontend."""
    print("Starting frontend server...")
    python_exe = get_python_executable()
    subprocess.Popen([
        python_exe, "-m", "streamlit", "run",
        "frontend/streamlit_app.py",
        "--server.port=8501"
    ])


def main():
    """Run both backend and frontend servers."""
    print("=" * 50)
    print("RAG Chatbot Application")
    print("=" * 50)
    print()
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set in .env file")
        print("   The application will not work without a valid OpenAI API key.")
        print()
    
    # Run backend and frontend
    try:
        print("Starting services...")
        print("-" * 50)
        run_backend()
        run_frontend()
        
        print()
        print("✅ Services started successfully!")
        print()
        print("📝 Backend API:    http://localhost:8000")
        print("   API Docs:       http://localhost:8000/docs")
        print()
        print("🤖 Chatbot UI:     http://localhost:8501")
        print()
        print("Press Ctrl+C to stop all services")
        print("-" * 50)
        
        # Keep the application running
        input()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
