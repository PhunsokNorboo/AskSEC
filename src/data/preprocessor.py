"""Document Preprocessor - Chunk documents for embedding."""
import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.data.parser import DocumentSection
from src.utils.config import settings


class DocumentChunker:
    """Chunk documents using semantic-aware splitting."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk (default from config)
            chunk_overlap: Overlap between chunks (default from config)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Separators ordered by priority (most to least preferred split points)
        self.separators = [
            "\n\n",      # Paragraph breaks (best)
            "\n",        # Line breaks
            ". ",        # Sentence endings
            "! ",
            "? ",
            "; ",
            ", ",
            " ",         # Words
            ""           # Characters (last resort)
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def _generate_chunk_id(self, ticker: str, item_number: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk."""
        unique_string = f"{ticker}_{item_number}_{chunk_index}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    def chunk_section(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Chunk a document section into smaller pieces.

        Args:
            content: Text content to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of LangChain Document objects
        """
        # Split the content
        chunks = self.text_splitter.split_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(
                metadata.get('ticker', 'UNK'),
                metadata.get('item_number', '0'),
                i
            )

            # Create metadata for this chunk
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
            }

            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))

        return documents

    def process_filing(
        self,
        sections: Dict[str, DocumentSection],
        ticker: str,
        filing_date: str,
        company_name: str
    ) -> List[Document]:
        """
        Process all sections of a filing into chunks.

        Args:
            sections: Dictionary of parsed sections
            ticker: Stock ticker
            filing_date: Date of filing
            company_name: Company name

        Returns:
            List of all Document chunks
        """
        all_documents = []

        for item_num, section in sections.items():
            # Base metadata for all chunks from this section
            metadata = {
                "ticker": ticker,
                "company_name": company_name,
                "filing_date": filing_date,
                "item_number": item_num,
                "item_title": section.item_title,
                "source": f"{ticker}_10K_{filing_date}",
                "section_length": len(section.content),
            }

            # Chunk this section
            chunks = self.chunk_section(section.content, metadata)
            all_documents.extend(chunks)

            print(f"    Item {item_num} ({section.item_title}): {len(chunks)} chunks")

        return all_documents

    def get_chunking_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the chunked documents."""
        if not documents:
            return {"total_chunks": 0}

        chunk_sizes = [len(doc.page_content) for doc in documents]

        # Group by company
        companies = {}
        for doc in documents:
            ticker = doc.metadata.get("ticker", "Unknown")
            if ticker not in companies:
                companies[ticker] = 0
            companies[ticker] += 1

        # Group by section
        sections = {}
        for doc in documents:
            section = doc.metadata.get("item_title", "Unknown")
            if section not in sections:
                sections[section] = 0
            sections[section] += 1

        return {
            "total_chunks": len(documents),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunks_by_company": companies,
            "chunks_by_section": sections,
        }
