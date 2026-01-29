"""SEC Filing Downloader - Download 10-K filings from SEC EDGAR."""
import os
import json
from typing import List, Dict
from edgar import Company, set_identity
from tqdm import tqdm

from src.utils.config import settings


class SECFilingDownloader:
    """Download 10-K filings from SEC EDGAR using edgartools."""

    def __init__(self, output_dir: str = None):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded filings.
                       Defaults to data/raw from config.
        """
        self.output_dir = output_dir or str(settings.RAW_DATA_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set identity for SEC EDGAR (required by SEC)
        set_identity(settings.EDGAR_IDENTITY)

    def download_10k_filings(
        self,
        tickers: List[str],
        num_filings: int = 2
    ) -> Dict[str, List[str]]:
        """
        Download recent 10-K filings for given tickers.

        Args:
            tickers: List of stock tickers (e.g., ["AAPL", "MSFT"])
            num_filings: Number of most recent 10-Ks to download per company

        Returns:
            Dictionary mapping ticker to list of saved file paths
        """
        results = {}

        for ticker in tqdm(tickers, desc="Downloading filings"):
            print(f"\n{'='*50}")
            print(f"Downloading 10-K filings for {ticker}...")
            print('='*50)

            try:
                # Get company info
                company = Company(ticker)
                print(f"Company: {company.name}")

                # Get 10-K filings
                filings = company.get_filings(form="10-K").head(num_filings)

                # Create directory for this ticker
                ticker_dir = os.path.join(self.output_dir, ticker)
                os.makedirs(ticker_dir, exist_ok=True)

                saved_files = []
                for filing in filings:
                    try:
                        # Get filing metadata
                        filing_date = str(filing.filing_date)
                        accession_no = filing.accession_number

                        print(f"  Processing filing from {filing_date}...")

                        # Get the text content (cleaner for RAG)
                        text_content = filing.text()

                        # Create filenames
                        base_name = f"{ticker}_10K_{filing_date}"
                        text_path = os.path.join(ticker_dir, f"{base_name}.txt")
                        meta_path = os.path.join(ticker_dir, f"{base_name}_meta.json")

                        # Save text content
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)

                        # Save metadata
                        metadata = {
                            "ticker": ticker,
                            "company_name": company.name,
                            "filing_date": filing_date,
                            "accession_number": accession_no,
                            "form_type": "10-K",
                            "cik": str(company.cik),
                            "file_path": text_path
                        }
                        with open(meta_path, 'w') as f:
                            json.dump(metadata, f, indent=2)

                        saved_files.append(text_path)
                        file_size = os.path.getsize(text_path) / 1024 / 1024  # MB
                        print(f"    Saved: {base_name}.txt ({file_size:.2f} MB)")

                    except Exception as e:
                        print(f"    Error processing filing: {e}")
                        continue

                results[ticker] = saved_files
                print(f"  Total: {len(saved_files)} filings downloaded for {ticker}")

            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")
                results[ticker] = []

        return results

    def get_download_summary(self, results: Dict[str, List[str]]) -> None:
        """Print a summary of downloaded filings."""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)

        total_files = 0
        total_size = 0

        for ticker, files in results.items():
            if files:
                ticker_size = sum(os.path.getsize(f) for f in files) / 1024 / 1024
                total_size += ticker_size
                total_files += len(files)
                print(f"{ticker}: {len(files)} filings ({ticker_size:.2f} MB)")
            else:
                print(f"{ticker}: No filings downloaded")

        print("-"*60)
        print(f"Total: {total_files} filings ({total_size:.2f} MB)")
        print("="*60)
