"""Tests for document preprocessor/chunker."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DocumentChunker
from src.data.parser import DocumentSection


class TestDocumentChunker:
    """Tests for the DocumentChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker with default settings."""
        return DocumentChunker(chunk_size=500, chunk_overlap=50)

    @pytest.fixture
    def sample_section(self):
        """Create a sample DocumentSection."""
        content = """
        This is a sample section with enough content to be chunked.
        It contains multiple paragraphs and sentences.

        The second paragraph discusses various topics related to
        business operations and financial performance.

        In the third paragraph, we explore risk factors and
        market conditions that may affect future results.
        """ * 10  # Repeat to ensure multiple chunks

        return DocumentSection(
            item_number="1A",
            item_title="Risk Factors",
            content=content,
            start_idx=0,
            end_idx=len(content)
        )

    def test_chunker_initialization(self, chunker):
        """Test chunker initializes with correct settings."""
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert len(chunker.separators) > 0

    def test_chunk_section_returns_documents(self, chunker, sample_section):
        """Test that chunking returns LangChain Documents."""
        metadata = {
            "ticker": "TEST",
            "item_number": "1A",
            "company_name": "Test Corp"
        }

        docs = chunker.chunk_section(sample_section.content, metadata)

        assert len(docs) > 0
        for doc in docs:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")

    def test_chunks_have_correct_metadata(self, chunker, sample_section):
        """Test that chunks preserve and extend metadata."""
        metadata = {
            "ticker": "TEST",
            "item_number": "1A"
        }

        docs = chunker.chunk_section(sample_section.content, metadata)

        for doc in docs:
            assert doc.metadata["ticker"] == "TEST"
            assert doc.metadata["item_number"] == "1A"
            assert "chunk_index" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert "total_chunks" in doc.metadata

    def test_chunk_ids_are_unique(self, chunker, sample_section):
        """Test that each chunk gets a unique ID."""
        metadata = {"ticker": "TEST", "item_number": "1"}

        docs = chunker.chunk_section(sample_section.content, metadata)
        chunk_ids = [doc.metadata["chunk_id"] for doc in docs]

        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_chunk_size_respected(self, chunker, sample_section):
        """Test that chunks respect the maximum size."""
        metadata = {"ticker": "TEST", "item_number": "1"}

        docs = chunker.chunk_section(sample_section.content, metadata)

        for doc in docs:
            # Allow some tolerance for chunk size
            assert len(doc.page_content) <= chunker.chunk_size + 100

    def test_process_filing_with_sections(self, chunker, sample_section):
        """Test processing a full filing with sections."""
        sections = {
            "1A": sample_section
        }

        docs = chunker.process_filing(
            sections=sections,
            ticker="TEST",
            filing_date="2025-01-15",
            company_name="Test Corp"
        )

        assert len(docs) > 0
        for doc in docs:
            assert doc.metadata["ticker"] == "TEST"
            assert doc.metadata["filing_date"] == "2025-01-15"
            assert doc.metadata["company_name"] == "Test Corp"

    def test_get_chunking_stats(self, chunker, sample_section):
        """Test chunking statistics generation."""
        metadata = {"ticker": "TEST", "item_number": "1"}
        docs = chunker.chunk_section(sample_section.content, metadata)

        stats = chunker.get_chunking_stats(docs)

        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats
        assert "min_chunk_size" in stats
        assert "max_chunk_size" in stats
        assert stats["total_chunks"] == len(docs)

    def test_empty_content_returns_empty_list(self, chunker):
        """Test that empty content returns empty list."""
        docs = chunker.chunk_section("", {"ticker": "TEST"})
        assert docs == []

    def test_short_content_single_chunk(self, chunker):
        """Test that short content results in single chunk."""
        short_content = "This is a very short piece of text."
        docs = chunker.chunk_section(short_content, {"ticker": "TEST"})

        assert len(docs) == 1
        assert docs[0].page_content == short_content

    def test_chunk_indices_are_sequential(self, chunker, sample_section):
        """Test that chunk indices are sequential starting from 0."""
        metadata = {"ticker": "TEST", "item_number": "1"}
        docs = chunker.chunk_section(sample_section.content, metadata)

        indices = [doc.metadata["chunk_index"] for doc in docs]
        assert indices == list(range(len(docs)))
