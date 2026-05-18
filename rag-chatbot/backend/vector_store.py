"""Vector store management module using FAISS and OpenAI embeddings."""

import os
import pickle
from typing import Optional, List
import numpy as np
import faiss
from openai import OpenAI
from backend.pdf_loader import Document


class VectorStore:
    """Manages FAISS vector store for document embeddings."""

    def __init__(self, store_path: str = "./faiss_db"):
        """Initialize the vector store.
        
        Args:
            store_path: Path to store/load FAISS index
        """
        self.store_path = store_path
        self.faiss_index: Optional[faiss.IndexIDMap] = None
        self.documents: List[Document] = []
        self.next_id = 0

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    @property
    def faiss_store(self) -> Optional[faiss.IndexIDMap]:
        """Compatibility alias for older code expecting faiss_store."""
        return self.faiss_index

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI embeddings."""
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)

        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not set. Cannot create embeddings.")

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(array)
        return array

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not set. Cannot create embeddings.")

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = np.array(response.data[0].embedding, dtype=np.float32)
        faiss.normalize_L2(query_vector.reshape(1, -1))
        return query_vector

    def create_from_documents(self, documents: List[Document]) -> None:
        """Create a new FAISS vector store from documents."""
        if not documents:
            raise ValueError("No documents provided to create vector store")

        print(f"Creating vector store from {len(documents)} documents...")
        self.documents = documents
        
        try:
            embeddings = self._embed_texts([doc.page_content for doc in documents])
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIDMap(index)

            ids = np.arange(len(documents), dtype=np.int64)
            self.faiss_index.add_with_ids(embeddings, ids)
            self.next_id = len(documents)
            print("Vector store created successfully using OpenAI Embeddings")
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e) or "insufficient" in str(e).lower():
                print("OpenAI API Quota exceeded. Initializing vector store in keyword-search fallback mode.")
                self.faiss_index = None
                self.next_id = len(documents)
            else:
                raise e

        self.save()
        print("Vector store saved successfully")

    def load(self) -> None:
        """Load existing FAISS index and documents from disk."""
        index_path = os.path.join(self.store_path, "index.faiss")
        docs_path = os.path.join(self.store_path, "documents.pkl")

        if os.path.exists(docs_path):
            print(f"Loading documents from {self.store_path}...")
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            self.next_id = len(self.documents)
            
            if os.path.exists(index_path):
                try:
                    self.faiss_index = faiss.read_index(index_path)
                    if not isinstance(self.faiss_index, faiss.IndexIDMap):
                        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
                    print("FAISS index loaded successfully")
                except Exception as e:
                    print(f"Could not load FAISS index: {str(e)}. Falling back to keyword search.")
                    self.faiss_index = None
            else:
                print("No FAISS index found. Loaded in keyword search fallback mode.")
                self.faiss_index = None
        else:
            print("No existing vector store found")

    def save(self) -> None:
        """Save the FAISS index and document metadata to disk."""
        print(f"Saving vector store to {self.store_path}...")
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(self.store_path, "index.faiss"))
        else:
            index_path = os.path.join(self.store_path, "index.faiss")
            if os.path.exists(index_path):
                try:
                    os.remove(index_path)
                except:
                    pass
        
        with open(os.path.join(self.store_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        print("Vector store saved successfully")

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of similar Document objects
        """
        if not self.documents:
            return []

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            # Keyword-based search fallback
            query_words = set(query.lower().split())
            scored_docs = []
            for doc in self.documents:
                content_lower = doc.page_content.lower()
                score = sum(1 for word in query_words if word in content_lower)
                scored_docs.append((score, doc))
            
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs[:k] if score > 0] or self.documents[:k]

        try:
            query_vector = self._embed_query(query)
            distances, ids = self.faiss_index.search(query_vector.reshape(1, -1), k)
            results = []

            for idx in ids[0]:
                if idx == -1:
                    continue
                results.append(self.documents[int(idx)])

            return results
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e) or "insufficient" in str(e).lower():
                print("OpenAI API Quota exceeded during search. Falling back to keyword search.")
                query_words = set(query.lower().split())
                scored_docs = []
                for doc in self.documents:
                    content_lower = doc.page_content.lower()
                    score = sum(1 for word in query_words if word in content_lower)
                    scored_docs.append((score, doc))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                return [doc for score, doc in scored_docs[:k] if score > 0] or self.documents[:k]
            else:
                raise e

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        if not documents:
            return

        if self.faiss_index is None:
            self.create_from_documents(documents)
            return

        print(f"Adding {len(documents)} documents to vector store...")
        try:
            embeddings = self._embed_texts([doc.page_content for doc in documents])
            ids = np.arange(self.next_id, self.next_id + len(documents), dtype=np.int64)
            self.faiss_index.add_with_ids(embeddings, ids)
            self.documents.extend(documents)
            self.next_id += len(documents)
            self.save()
            print("Documents added and saved successfully using OpenAI Embeddings")
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e) or "insufficient" in str(e).lower():
                print("OpenAI API Quota exceeded during document addition. Appending documents in keyword-search fallback mode.")
                self.documents.extend(documents)
                self.next_id += len(documents)
                self.save()
            else:
                raise e
