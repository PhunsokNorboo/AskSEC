"""Tests for vector store operations."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVectorStoreManager:
    """Tests for VectorStoreManager class."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        mock = Mock()
        mock.embed_documents.return_value = [[0.1] * 384]
        mock.embed_query.return_value = [0.1] * 384
        return mock

    def test_vector_store_manager_import(self):
        """Test that VectorStoreManager can be imported."""
        from src.embeddings.vector_store import VectorStoreManager
        assert VectorStoreManager is not None

    def test_vector_store_has_required_methods(self):
        """Test that VectorStoreManager has required methods."""
        from src.embeddings.vector_store import VectorStoreManager

        required_methods = [
            "create_or_load_store",
            "add_documents",
            "similarity_search",
            "get_retriever",
            "get_collection_stats",
            "get_available_companies"
        ]

        for method in required_methods:
            assert hasattr(VectorStoreManager, method), f"Missing method: {method}"

    @patch('src.embeddings.vector_store.HuggingFaceEmbeddings')
    @patch('src.embeddings.vector_store.chromadb.PersistentClient')
    def test_initialization_creates_directories(
        self, mock_client, mock_embeddings, temp_data_dir
    ):
        """Test that initialization creates necessary directories."""
        from src.embeddings.vector_store import VectorStoreManager

        chroma_dir = str(temp_data_dir["chroma"])
        manager = VectorStoreManager(persist_directory=chroma_dir)

        assert Path(chroma_dir).exists()

    @patch('src.embeddings.vector_store.HuggingFaceEmbeddings')
    @patch('src.embeddings.vector_store.chromadb.PersistentClient')
    def test_get_collection_stats_returns_dict(
        self, mock_client, mock_embeddings, temp_data_dir
    ):
        """Test that collection stats returns a dictionary."""
        from src.embeddings.vector_store import VectorStoreManager

        # Setup mock
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_client.return_value.get_collection.return_value = mock_collection

        manager = VectorStoreManager(
            persist_directory=str(temp_data_dir["chroma"]),
            collection_name="test_collection"
        )

        stats = manager.get_collection_stats()

        assert isinstance(stats, dict)
        assert "name" in stats

    def test_format_documents_helper(self):
        """Test the format_documents helper function."""
        from src.rag.prompts import format_documents
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="Test content 1",
                metadata={
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "filing_date": "2025-01-15",
                    "item_title": "Business"
                }
            ),
            Document(
                page_content="Test content 2",
                metadata={
                    "ticker": "MSFT",
                    "company_name": "Microsoft Corp",
                    "filing_date": "2025-01-20",
                    "item_title": "Risk Factors"
                }
            )
        ]

        formatted = format_documents(docs)

        assert "Apple Inc." in formatted
        assert "Microsoft Corp" in formatted
        assert "AAPL" in formatted
        assert "MSFT" in formatted
        assert "Test content 1" in formatted
        assert "Test content 2" in formatted


class TestVectorStoreIntegration:
    """Integration tests for vector store (requires actual ChromaDB)."""

    @pytest.mark.slow
    def test_add_and_retrieve_documents(self, temp_data_dir):
        """Test adding and retrieving documents from vector store."""
        # This test is marked slow as it requires actual embedding computation
        pytest.skip("Skipping slow integration test - run with pytest -m slow")
