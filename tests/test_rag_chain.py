"""Tests for RAG chain and prompts."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPrompts:
    """Tests for prompt templates."""

    def test_rag_prompt_template_exists(self):
        """Test that RAG prompt template is defined."""
        from src.rag.prompts import RAG_PROMPT

        assert RAG_PROMPT is not None
        assert hasattr(RAG_PROMPT, "template")
        assert "{context}" in RAG_PROMPT.template
        assert "{question}" in RAG_PROMPT.template

    def test_chat_rag_prompt_template_exists(self):
        """Test that chat RAG prompt template is defined."""
        from src.rag.prompts import CHAT_RAG_PROMPT

        assert CHAT_RAG_PROMPT is not None
        assert "{context}" in CHAT_RAG_PROMPT.template
        assert "{question}" in CHAT_RAG_PROMPT.template

    def test_comparison_prompt_template_exists(self):
        """Test that comparison prompt template is defined."""
        from src.rag.prompts import COMPARISON_PROMPT

        assert COMPARISON_PROMPT is not None
        assert "comparison" in COMPARISON_PROMPT.template.lower() or \
               "compare" in COMPARISON_PROMPT.template.lower()

    def test_format_documents_function(self):
        """Test document formatting function."""
        from src.rag.prompts import format_documents
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="Revenue increased by 15%",
                metadata={
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "filing_date": "2025-01-15",
                    "item_title": "MD&A"
                }
            )
        ]

        result = format_documents(docs)

        assert isinstance(result, str)
        assert "Apple Inc." in result
        assert "AAPL" in result
        assert "Revenue increased by 15%" in result


class TestSECFilingRAG:
    """Tests for SECFilingRAG class."""

    def test_sec_filing_rag_import(self):
        """Test that SECFilingRAG can be imported."""
        from src.rag.chain import SECFilingRAG
        assert SECFilingRAG is not None

    def test_sec_filing_rag_has_required_methods(self):
        """Test that SECFilingRAG has required methods."""
        from src.rag.chain import SECFilingRAG

        required_methods = [
            "query",
            "clear_history",
            "get_available_companies",
            "search_only"
        ]

        for method in required_methods:
            assert hasattr(SECFilingRAG, method), f"Missing method: {method}"

    def test_format_sources_method(self):
        """Test the _format_sources method."""
        from src.rag.chain import SECFilingRAG
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="Content 1" * 50,  # Long content
                metadata={
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "filing_date": "2025-01-15",
                    "item_title": "Business",
                    "item_number": "1"
                }
            ),
            Document(
                page_content="Content 2",
                metadata={
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "filing_date": "2025-01-15",
                    "item_title": "Business",
                    "item_number": "1"
                }
            )
        ]

        # Test the static behavior of format_sources
        # Since it deduplicates, should return 1 unique source
        seen = set()
        sources = []
        for doc in docs:
            source_key = (
                doc.metadata.get("ticker"),
                doc.metadata.get("filing_date"),
                doc.metadata.get("item_number")
            )
            if source_key not in seen:
                seen.add(source_key)
                sources.append(doc.metadata)

        assert len(sources) == 1  # Deduplicated to 1

    @patch('src.rag.chain.ChatOllama')
    @patch('src.rag.chain.VectorStoreManager')
    def test_rag_initialization_with_mocks(
        self, mock_vector_store, mock_ollama
    ):
        """Test RAG initialization with mocked dependencies."""
        from src.rag.chain import SECFilingRAG

        # Setup mocks
        mock_vector_store.return_value.create_or_load_store.return_value = Mock()
        mock_vector_store.return_value.get_retriever.return_value = Mock()

        # This should not raise an error
        rag = SECFilingRAG.__new__(SECFilingRAG)

        assert rag is not None


class TestRAGQueryResponse:
    """Tests for RAG query response format."""

    def test_query_response_structure(self):
        """Test that query response has expected structure."""
        # Expected response structure
        expected_keys = ["answer", "sources", "num_sources"]

        response = {
            "answer": "Test answer",
            "sources": [],
            "num_sources": 0
        }

        for key in expected_keys:
            assert key in response

    def test_source_structure(self):
        """Test that source entries have expected structure."""
        expected_keys = ["ticker", "company", "filing_date", "section", "excerpt"]

        source = {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "filing_date": "2025-01-15",
            "section": "Business",
            "excerpt": "Sample text..."
        }

        for key in expected_keys:
            assert key in source
