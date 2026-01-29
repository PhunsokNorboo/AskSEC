#!/usr/bin/env python3
"""Script to process downloaded SEC filings into chunks."""
import sys
import os
import json
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.parser import SEC10KParser
from src.data.preprocessor import DocumentChunker
from src.utils.config import settings


def process_all_filings():
    """Process all downloaded filings into chunks."""

    print("="*60)
    print("SEC 10-K Filing Processor")
    print("="*60)

    # Initialize parser and chunker
    parser = SEC10KParser()
    chunker = DocumentChunker()

    print(f"\nChunk settings:")
    print(f"  - Chunk size: {chunker.chunk_size} characters")
    print(f"  - Chunk overlap: {chunker.chunk_overlap} characters")

    raw_data_dir = settings.RAW_DATA_DIR
    processed_dir = settings.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_documents = []
    processing_stats = {}

    # Get list of company folders
    company_folders = [d for d in raw_data_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(company_folders)} companies to process\n")

    for ticker_dir in sorted(company_folders):
        ticker = ticker_dir.name
        print(f"{'='*50}")
        print(f"Processing {ticker}...")
        print('='*50)

        ticker_docs = []

        # Find all text files for this company
        text_files = list(ticker_dir.glob("*_10K_*.txt"))

        for text_path in sorted(text_files):
            # Load metadata
            meta_path = str(text_path).replace('.txt', '_meta.json')
            if not os.path.exists(meta_path):
                print(f"  Warning: No metadata for {text_path.name}")
                continue

            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # Load text content
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

            print(f"\n  Filing: {metadata['filing_date']}")
            print(f"  Original size: {len(text):,} characters")

            # Parse sections
            sections = parser.parse_filing(text)
            print(f"  Sections found: {len(sections)}")

            if not sections:
                print("  Warning: No sections extracted!")
                continue

            # Show section summary
            for item_num, section in sections.items():
                print(f"    - Item {item_num} ({section.item_title}): {len(section):,} chars")

            # Chunk the filing
            print(f"\n  Chunking...")
            documents = chunker.process_filing(
                sections=sections,
                ticker=metadata['ticker'],
                filing_date=metadata['filing_date'],
                company_name=metadata['company_name']
            )

            ticker_docs.extend(documents)
            print(f"  Total chunks for this filing: {len(documents)}")

        all_documents.extend(ticker_docs)
        processing_stats[ticker] = len(ticker_docs)
        print(f"\n  {ticker} total: {len(ticker_docs)} chunks")

    # Save processed documents
    output_path = processed_dir / "all_documents.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_documents, f)

    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print('='*60)

    # Get detailed stats
    stats = chunker.get_chunking_stats(all_documents)

    print(f"\nTotal chunks: {stats['total_chunks']}")
    print(f"Average chunk size: {stats['avg_chunk_size']:.0f} characters")
    print(f"Min chunk size: {stats['min_chunk_size']} characters")
    print(f"Max chunk size: {stats['max_chunk_size']} characters")

    print(f"\nChunks by company:")
    for company, count in sorted(stats['chunks_by_company'].items()):
        print(f"  {company}: {count}")

    print(f"\nChunks by section:")
    for section, count in sorted(stats['chunks_by_section'].items(), key=lambda x: -x[1]):
        print(f"  {section}: {count}")

    print(f"\nProcessed documents saved to: {output_path}")
    print('='*60)

    return all_documents


if __name__ == "__main__":
    documents = process_all_filings()
