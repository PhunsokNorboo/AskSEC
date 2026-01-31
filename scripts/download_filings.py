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

    # Target companies - diverse mix across sectors
    tickers = [
        # Technology (6)
        "AAPL",   # Apple - Consumer Electronics
        "MSFT",   # Microsoft - Software/Cloud
        "GOOGL",  # Alphabet - Search/Advertising
        "META",   # Meta - Social Media
        "NVDA",   # Nvidia - Semiconductors/AI
        "AMZN",   # Amazon - E-commerce/Cloud

        # Automotive (1)
        "TSLA",   # Tesla - Electric Vehicles

        # Entertainment & Retail (3)
        "NFLX",   # Netflix - Streaming
        "WMT",    # Walmart - Retail
        "MCD",    # McDonald's - Fast Food

        # Fintech & Payments (2)
        "V",      # Visa - Payments
        "PYPL",   # PayPal - Digital Payments

        # Healthcare & Pharma (3)
        "JNJ",    # Johnson & Johnson - Healthcare
        "PFE",    # Pfizer - Pharmaceuticals
        "MRK",    # Merck - Pharmaceuticals

        # Banks & Finance (3)
        "JPM",    # JPMorgan Chase - Banking
        "BAC",    # Bank of America - Banking
        "GS",     # Goldman Sachs - Investment Banking

        # Energy (2)
        "XOM",    # ExxonMobil - Oil & Gas
        "CVX",    # Chevron - Oil & Gas

        # Industrials (2)
        "CAT",    # Caterpillar - Heavy Equipment
        "BA",     # Boeing - Aerospace

        # Consumer Goods (1)
        "KO",     # Coca-Cola - Beverages
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
