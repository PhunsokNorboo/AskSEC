#!/usr/bin/env python3
"""Script to build the vector database from processed documents."""
import sys
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.vector_store import VectorStoreManager
from src.utils.config import settings


def build_vector_database():
    """Build the vector database from processed documents."""

    print("="*60)
    print("Vector Database Builder")
    print("="*60)

    # Load processed documents
    processed_path = settings.PROCESSED_DATA_DIR / "all_documents.pkl"

    if not processed_path.exists():
        print(f"\nError: Processed documents not found at {processed_path}")
        print("Please run scripts/process_filings.py first.")
        sys.exit(1)

    print(f"\nLoading processed documents from: {processed_path}")
    with open(processed_path, 'rb') as f:
        documents = pickle.load(f)

    print(f"Loaded {len(documents)} documents")

    # Initialize vector store
    print(f"\nInitializing vector store...")
    print(f"  Persist directory: {settings.CHROMA_DB_DIR}")
    print(f"  Collection name: {settings.CHROMA_COLLECTION_NAME}")
    print(f"  Embedding model: {settings.EMBEDDING_MODEL}")

    vector_store = VectorStoreManager()

    # Check if collection already exists
    stats = vector_store.get_collection_stats()
    if stats.get("count", 0) > 0:
        print(f"\nExisting collection found with {stats['count']} documents.")
        response = input("Delete and rebuild? (y/n): ").strip().lower()
        if response == 'y':
            vector_store.delete_collection()
        else:
            print("Keeping existing collection. Exiting.")
            return

    # Create vector store and add documents
    vector_store.create_or_load_store()
    vector_store.add_documents(documents, batch_size=100)

    # Get final stats
    stats = vector_store.get_collection_stats()

    print(f"\n{'='*60}")
    print("BUILD COMPLETE")
    print('='*60)
    print(f"\nCollection: {stats['name']}")
    print(f"Total documents: {stats['count']}")
    print(f"Storage location: {stats['persist_directory']}")

    # Test a sample query
    print(f"\n{'='*60}")
    print("Testing vector search...")
    print('='*60)

    test_queries = [
        "What are the main risk factors?",
        "How does the company generate revenue?",
        "What is the competitive landscape?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            ticker = doc.metadata.get('ticker', 'N/A')
            section = doc.metadata.get('item_title', 'N/A')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i}. [{ticker}] {section}")
            print(f"     {preview}...")

    # List available companies
    companies = vector_store.get_available_companies()
    print(f"\nCompanies in database: {', '.join(companies)}")

    print(f"\n{'='*60}")
    print("Vector database ready for RAG queries!")
    print('='*60)


if __name__ == "__main__":
    build_vector_database()
