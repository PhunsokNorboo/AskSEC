"""Integration tests for AskSEC pipeline."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parser import SEC10KParser
from src.data.preprocessor import DocumentChunker


class TestParserPreprocessorIntegration:
    """Test parser and preprocessor work together correctly."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return SEC10KParser()

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return DocumentChunker(chunk_size=500, chunk_overlap=50)

    @pytest.fixture
    def long_10k_text(self):
        """Sample 10-K with sections long enough to be extracted (>500 chars)."""
        return """
        ITEM 1. BUSINESS

        We are a leading technology company specializing in consumer electronics,
        software, and digital services. Our products include smartphones, tablets,
        personal computers, wearables, and accessories. We also provide a range of
        services including cloud storage, music streaming, app distribution, and
        digital content. Our company was founded in 1976 and has grown to become
        one of the largest technology companies in the world with operations in
        over 100 countries. We employ approximately 150,000 full-time employees
        worldwide and have an extensive network of retail stores, online channels,
        and authorized resellers. Our commitment to innovation drives our continued
        investment in research and development.

        ITEM 1A. RISK FACTORS

        Investing in our securities involves significant risks. Prospective investors
        should carefully consider the following risk factors before making an investment
        decision. Competition in our industry is intense and includes many large and
        well-established companies with greater resources. Economic conditions affect
        consumer spending patterns and demand for our products. Supply chain disruptions
        could impact our ability to manufacture and deliver products on time. Currency
        fluctuations affect our international operations and financial results.
        Regulatory changes in various jurisdictions could increase compliance costs.
        Cybersecurity threats pose risks to our data and systems. Technology changes
        rapidly and we must continue to innovate to remain competitive.
        """

    def test_full_parsing_and_chunking_pipeline(
        self, parser, chunker, long_10k_text, sample_metadata
    ):
        """Test parsing a 10-K and chunking the sections."""
        # Parse the filing
        sections = parser.parse_filing(long_10k_text)

        # Process through chunker
        documents = chunker.process_filing(
            sections=sections,
            ticker=sample_metadata["ticker"],
            filing_date=sample_metadata["filing_date"],
            company_name=sample_metadata["company_name"],
        )

        # Verify results
        assert len(documents) > 0

        # Check all documents have required metadata
        for doc in documents:
            assert "ticker" in doc.metadata
            assert "company_name" in doc.metadata
            assert "filing_date" in doc.metadata
            assert "item_number" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert doc.page_content  # Content is not empty

    def test_metadata_preservation_through_pipeline(
        self, parser, chunker, sample_10k_text, sample_metadata
    ):
        """Test that metadata is correctly preserved through the pipeline."""
        sections = parser.parse_filing(sample_10k_text)
        documents = chunker.process_filing(
            sections=sections,
            ticker="AAPL",
            filing_date="2025-01-15",
            company_name="Apple Inc.",
        )

        # Verify ticker is preserved
        for doc in documents:
            assert doc.metadata["ticker"] == "AAPL"
            assert doc.metadata["company_name"] == "Apple Inc."
            assert doc.metadata["filing_date"] == "2025-01-15"

    def test_chunk_ids_are_unique(self, parser, chunker, sample_10k_text):
        """Test that all chunk IDs are unique."""
        sections = parser.parse_filing(sample_10k_text)
        documents = chunker.process_filing(
            sections=sections,
            ticker="TEST",
            filing_date="2025-01-15",
            company_name="Test Company",
        )

        chunk_ids = [doc.metadata["chunk_id"] for doc in documents]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

    def test_empty_filing_produces_no_documents(self, parser, chunker):
        """Test that empty filing produces no documents."""
        sections = parser.parse_filing("")
        documents = chunker.process_filing(
            sections=sections,
            ticker="TEST",
            filing_date="2025-01-15",
            company_name="Test Company",
        )

        assert len(documents) == 0


class TestRAGChainIntegration:
    """Test RAG chain integration with mocked dependencies."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store manager."""
        mock = MagicMock()
        mock.get_available_companies.return_value = ["AAPL", "MSFT", "GOOGL"]
        mock.create_or_load_store.return_value = MagicMock()

        # Mock retriever
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content about Apple's business."
        mock_doc.metadata = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_date": "2025-01-15",
            "item_title": "Business",
        }
        mock_retriever.invoke.return_value = [mock_doc]
        mock.get_retriever.return_value = mock_retriever

        return mock

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Apple is a technology company."
        mock.invoke.return_value = mock_response
        return mock

    @patch("src.rag.chain.VectorStoreManager")
    @patch("src.rag.chain.ChatOllama")
    def test_rag_query_returns_expected_structure(
        self, mock_ollama_class, mock_vs_class, mock_vector_store, mock_llm
    ):
        """Test RAG query returns expected response structure."""
        mock_vs_class.return_value = mock_vector_store
        mock_ollama_class.return_value = mock_llm

        from src.rag.chain import SECFilingRAG

        rag = SECFilingRAG()
        result = rag.query("What does Apple do?")

        assert "answer" in result
        assert "sources" in result
        assert "num_sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

    @patch("src.rag.chain.VectorStoreManager")
    @patch("src.rag.chain.ChatOllama")
    def test_rag_validates_invalid_ticker(
        self, mock_ollama_class, mock_vs_class, mock_vector_store, mock_llm
    ):
        """Test RAG validates ticker format."""
        mock_vs_class.return_value = mock_vector_store
        mock_ollama_class.return_value = mock_llm

        from src.rag.chain import SECFilingRAG

        rag = SECFilingRAG()

        with pytest.raises(ValueError, match="Invalid ticker format"):
            rag.query("Test question", filter_ticker="INVALID123")

    @patch("src.rag.chain.VectorStoreManager")
    @patch("src.rag.chain.ChatOllama")
    def test_rag_validates_empty_question(
        self, mock_ollama_class, mock_vs_class, mock_vector_store, mock_llm
    ):
        """Test RAG validates empty questions."""
        mock_vs_class.return_value = mock_vector_store
        mock_ollama_class.return_value = mock_llm

        from src.rag.chain import SECFilingRAG

        rag = SECFilingRAG()

        with pytest.raises(ValueError, match="Question cannot be empty"):
            rag.query("")

    @patch("src.rag.chain.VectorStoreManager")
    @patch("src.rag.chain.ChatOllama")
    def test_rag_validates_long_question(
        self, mock_ollama_class, mock_vs_class, mock_vector_store, mock_llm
    ):
        """Test RAG validates question length."""
        mock_vs_class.return_value = mock_vector_store
        mock_ollama_class.return_value = mock_llm

        from src.rag.chain import SECFilingRAG

        rag = SECFilingRAG()
        long_question = "a" * 3000

        with pytest.raises(ValueError, match="Question too long"):
            rag.query(long_question)

    @patch("src.rag.chain.VectorStoreManager")
    @patch("src.rag.chain.ChatOllama")
    def test_rag_accepts_valid_ticker(
        self, mock_ollama_class, mock_vs_class, mock_vector_store, mock_llm
    ):
        """Test RAG accepts valid ticker symbols."""
        mock_vs_class.return_value = mock_vector_store
        mock_ollama_class.return_value = mock_llm

        from src.rag.chain import SECFilingRAG

        rag = SECFilingRAG()

        # These should not raise
        result = rag.query("Test question", filter_ticker="AAPL")
        assert result is not None

        result = rag.query("Test question", filter_ticker="msft")  # lowercase OK
        assert result is not None

    @patch("src.rag.chain.VectorStoreManager")
    @patch("src.rag.chain.ChatOllama")
    def test_rag_chat_history_updates(
        self, mock_ollama_class, mock_vs_class, mock_vector_store, mock_llm
    ):
        """Test chat history is updated after query."""
        mock_vs_class.return_value = mock_vector_store
        mock_ollama_class.return_value = mock_llm

        from src.rag.chain import SECFilingRAG

        rag = SECFilingRAG()
        assert len(rag.chat_history) == 0

        rag.query("First question")
        assert len(rag.chat_history) == 1

        rag.query("Second question")
        assert len(rag.chat_history) == 2

        rag.clear_history()
        assert len(rag.chat_history) == 0


class TestDataFlowIntegration:
    """Test data flows correctly through the entire system."""

    @pytest.fixture
    def long_10k_for_data_flow(self):
        """Long sample for data flow tests."""
        return """
        ITEM 1. BUSINESS

        We are a leading technology company specializing in consumer electronics,
        software, and digital services. Our products include smartphones, tablets,
        personal computers, wearables, and accessories. We also provide a range of
        services including cloud storage, music streaming, app distribution, and
        digital content. Our company was founded in 1976 and has grown to become
        one of the largest technology companies in the world with operations in
        over 100 countries. We employ approximately 150,000 full-time employees
        worldwide and have an extensive network of retail stores and resellers.

        ITEM 1A. RISK FACTORS

        Investing in our securities involves significant risks. Prospective investors
        should carefully consider the following risk factors before making investment.
        Competition in our industry is intense and includes many large companies.
        Economic conditions affect consumer spending patterns and demand for products.
        Supply chain disruptions could impact our ability to manufacture and deliver.
        Currency fluctuations affect our international operations and results.
        Regulatory changes in various jurisdictions could increase compliance costs.
        """

    def test_section_titles_preserved_in_chunks(self, long_10k_for_data_flow):
        """Test that section titles are preserved through chunking."""
        parser = SEC10KParser()
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

        sections = parser.parse_filing(long_10k_for_data_flow)
        documents = chunker.process_filing(
            sections=sections,
            ticker="TEST",
            filing_date="2025-01-15",
            company_name="Test Company",
        )

        # Get unique section titles
        section_titles = {doc.metadata.get("item_title") for doc in documents}

        # Should have at least one section
        assert len(section_titles) > 0

        # Titles should be human-readable
        for title in section_titles:
            assert title is not None
            assert len(title) > 0

    def test_chunking_stats_are_accurate(self, long_10k_for_data_flow):
        """Test chunking statistics are accurate."""
        parser = SEC10KParser()
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

        sections = parser.parse_filing(long_10k_for_data_flow)
        documents = chunker.process_filing(
            sections=sections,
            ticker="TEST",
            filing_date="2025-01-15",
            company_name="Test Company",
        )

        stats = chunker.get_chunking_stats(documents)

        assert stats["total_chunks"] == len(documents)
        assert stats["avg_chunk_size"] > 0
        assert stats["min_chunk_size"] <= stats["max_chunk_size"]
        assert "TEST" in stats["chunks_by_company"]
