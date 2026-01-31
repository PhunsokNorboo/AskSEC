"""Vector Store Manager - ChromaDB operations with sentence-transformers."""
import logging
import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.utils.config import settings

logger = logging.getLogger("asksec.vector_store")


class VectorStoreManager:
    """Manage Chroma vector database operations with local embeddings."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None
    ) -> None:
        """
        Initialize the vector store manager.

        Args:
            persist_directory: Directory to store the vector database
            collection_name: Name of the collection
            embedding_model: Sentence-transformers model name
        """
        self.persist_directory = persist_directory or str(settings.CHROMA_DB_DIR)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL

        # Initialize embeddings (local, free)
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use 'mps' for Apple Silicon GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded successfully")

        # Initialize Chroma client
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.vector_store: Chroma | None = None

    def create_or_load_store(self) -> Chroma:
        """Create a new vector store or load existing one."""
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        return self.vector_store

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store in batches.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process at once
        """
        if self.vector_store is None:
            self.create_or_load_store()

        total_docs = len(documents)
        total_batches = (total_docs + batch_size - 1) // batch_size

        print(f"\nAdding {total_docs} documents in {total_batches} batches...")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Extract IDs from metadata
            ids = [doc.metadata.get("chunk_id", f"doc_{i+j}") for j, doc in enumerate(batch)]

            # Add batch to vector store
            self.vector_store.add_documents(
                documents=batch,
                ids=ids
            )

            # Progress update
            progress = (batch_num / total_batches) * 100
            print(f"  Batch {batch_num}/{total_batches} ({progress:.1f}%) - {len(batch)} documents")

        print(f"\nSuccessfully added {total_docs} documents to vector store")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"ticker": "AAPL"})

        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            self.create_or_load_store()

        if filter_dict:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None
    ) -> list[tuple]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            self.create_or_load_store()

        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

        return results

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Perform MMR (Maximal Marginal Relevance) search for diverse results.

        MMR balances relevance with diversity, reducing redundant results.

        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of candidates to fetch before reranking
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance)
            filter_dict: Optional metadata filters

        Returns:
            List of diverse, relevant documents
        """
        if self.vector_store is None:
            self.create_or_load_store()

        if filter_dict:
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_dict
            )
        else:
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )

        return results

    def get_retriever(
        self,
        search_kwargs: dict[str, Any] | None = None,
        use_mmr: bool = True
    ) -> Any:
        """
        Get a retriever for use with LangChain chains.

        Args:
            search_kwargs: Search parameters (e.g., {"k": 5, "filter": {...}})
            use_mmr: If True, use MMR for diverse results (recommended)

        Returns:
            LangChain retriever object
        """
        if self.vector_store is None:
            self.create_or_load_store()

        search_kwargs = search_kwargs or {"k": settings.RETRIEVAL_K}

        if use_mmr:
            # MMR provides more diverse results
            search_kwargs.setdefault("fetch_k", search_kwargs.get("k", 6) * 3)
            search_kwargs.setdefault("lambda_mult", 0.5)
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
        else:
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_directory": self.persist_directory,
            }
        except ValueError as e:
            logger.warning(f"Collection '{self.collection_name}' not found: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "error": str(e)
            }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.vector_store = None
            logger.info(f"Collection '{self.collection_name}' deleted")
        except ValueError as e:
            logger.warning(f"Collection not found for deletion: {e}")

    def get_available_companies(self) -> list[str]:
        """Get list of unique company tickers in the database."""
        if self.vector_store is None:
            self.create_or_load_store()

        try:
            collection = self.client.get_collection(self.collection_name)
            # Get sample of metadata
            results = collection.get(limit=10000, include=["metadatas"])
            tickers = set()
            for meta in results.get("metadatas", []):
                if meta and "ticker" in meta:
                    tickers.add(meta["ticker"])
            return sorted(list(tickers))
        except ValueError as e:
            logger.warning(f"Could not retrieve companies: {e}")
            return []
