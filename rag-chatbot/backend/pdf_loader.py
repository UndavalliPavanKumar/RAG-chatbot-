"""PDF document loader module for the RAG chatbot."""

import os
from pathlib import Path
from typing import List
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class Document:
    """Lightweight document container."""
    page_content: str
    metadata: dict


class PDFLoader:
    """Loads and processes PDF documents."""

    def __init__(self, data_dir: str = "./data"):
        """Initialize the PDF loader.
        
        Args:
            data_dir: Directory containing PDF files
        """
        self.data_dir = data_dir

    def load_pdfs(self) -> List[Document]:
        """Load all PDF files from the data directory.
        
        Returns:
            List of Document objects
        """
        documents = []
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            return documents
        
        for pdf_file in Path(self.data_dir).glob("*.pdf"):
            try:
                documents.extend(self._load_single_pdf(str(pdf_file)))
            except Exception as e:
                print(f"Error loading {pdf_file}: {str(e)}")
        
        return documents

    def _load_single_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects extracted from the PDF
        """
        documents = []
        
        try:
            pdf_reader = PdfReader(file_path)
            file_name = os.path.basename(file_path)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_num + 1,
                        "file_path": file_path
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            raise Exception(f"Failed to load PDF {file_path}: {str(e)}")
        
        return documents
