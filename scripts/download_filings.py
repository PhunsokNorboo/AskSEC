#!/usr/bin/env python3
"""Script to download SEC 10-K filings."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import SECFilingDownloader
from src.utils.config import settings


def main():
    """Download 10-K filings for target companies."""

    # Validate configuration
    if not settings.validate():
        print("\nPlease configure your .env file before running.")
        print("See .env.example for required settings.")
        sys.exit(1)

    # Ensure data directories exist
    settings.ensure_directories()

    # Target companies - mix of tech, finance, healthcare, retail
    tickers = [
        "AAPL",   # Apple - Tech
        "MSFT",   # Microsoft - Tech
        "TSLA",   # Tesla - Automotive/Tech
        "AMZN",   # Amazon - E-commerce/Cloud
        "GOOGL",  # Alphabet - Tech
        "META",   # Meta - Social Media/Tech
        "NVDA",   # Nvidia - Semiconductors
        "JPM",    # JPMorgan Chase - Finance
        "JNJ",    # Johnson & Johnson - Healthcare
        "WMT",    # Walmart - Retail
    ]

    print("="*60)
    print("SEC 10-K Filing Downloader")
    print("="*60)
    print(f"\nTarget companies: {', '.join(tickers)}")
    print(f"Filings per company: 2 (most recent)")
    print(f"Output directory: {settings.RAW_DATA_DIR}")
    print(f"EDGAR Identity: {settings.EDGAR_IDENTITY}")
    print()

    # Initialize downloader
    downloader = SECFilingDownloader()

    # Download filings
    results = downloader.download_10k_filings(
        tickers=tickers,
        num_filings=2  # Last 2 years of 10-K filings
    )

    # Print summary
    downloader.get_download_summary(results)

    print("\nDownload complete! Files saved to:", settings.RAW_DATA_DIR)


if __name__ == "__main__":
    main()
