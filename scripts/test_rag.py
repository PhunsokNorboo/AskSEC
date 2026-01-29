#!/usr/bin/env python3
"""Script to test the RAG pipeline."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.chain import SECFilingRAG


def test_rag_pipeline():
    """Test the RAG pipeline with sample questions."""

    print("="*60)
    print("SEC Filing RAG Pipeline Test")
    print("="*60)

    # Initialize RAG
    print("\nInitializing RAG pipeline...")
    rag = SECFilingRAG()

    # Get available companies
    companies = rag.get_available_companies()
    print(f"\nCompanies in database: {', '.join(companies)}")

    # Test questions
    test_questions = [
        # General questions
        ("What are Apple's main business segments?", None),
        ("What are the key risk factors for Tesla?", "TSLA"),
        ("How does Netflix generate revenue?", "NFLX"),

        # Comparison question
        ("Compare the competitive advantages of Nvidia and Meta", None),

        # Financial question
        ("What does Visa say about their payment volume growth?", "V"),
    ]

    print("\n" + "="*60)
    print("Running test queries...")
    print("="*60)

    for question, filter_ticker in test_questions:
        print(f"\n{'─'*60}")
        if filter_ticker:
            print(f"Question (filtered to {filter_ticker}): {question}")
        else:
            print(f"Question: {question}")
        print('─'*60)

        # Query the RAG system
        result = rag.query(question, filter_ticker=filter_ticker)

        # Display answer
        print(f"\nAnswer:\n{result['answer']}")

        # Display sources
        print(f"\nSources ({result['num_sources']} documents):")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source['company']} ({source['ticker']})")
            print(f"     Filed: {source['filing_date']} | Section: {source['section']}")

    print("\n" + "="*60)
    print("RAG Pipeline Test Complete!")
    print("="*60)


def interactive_mode():
    """Run interactive Q&A session."""

    print("="*60)
    print("SEC Filing RAG - Interactive Mode")
    print("="*60)

    # Initialize RAG
    print("\nInitializing RAG pipeline...")
    rag = SECFilingRAG()

    companies = rag.get_available_companies()
    print(f"\nAvailable companies: {', '.join(companies)}")
    print("\nTips:")
    print("  - Type your question and press Enter")
    print("  - Type 'filter TICKER' to filter by company (e.g., 'filter AAPL')")
    print("  - Type 'clear' to clear filter")
    print("  - Type 'quit' to exit")

    current_filter = None

    while True:
        print("\n" + "-"*40)
        if current_filter:
            prompt = f"[Filter: {current_filter}] Your question: "
        else:
            prompt = "Your question: "

        try:
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            current_filter = None
            print("Filter cleared.")
            continue

        if user_input.lower().startswith('filter '):
            ticker = user_input.split(' ', 1)[1].upper()
            if ticker in companies:
                current_filter = ticker
                print(f"Filter set to: {current_filter}")
            else:
                print(f"Unknown ticker: {ticker}")
                print(f"Available: {', '.join(companies)}")
            continue

        # Query RAG
        print("\nSearching and generating answer...")
        result = rag.query(user_input, filter_ticker=current_filter)

        print(f"\n{result['answer']}")

        print(f"\n[Sources: {result['num_sources']} documents from: ", end="")
        tickers = list(set(s['ticker'] for s in result['sources']))
        print(f"{', '.join(tickers)}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        test_rag_pipeline()
