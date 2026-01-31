"""Tests for SEC Filing Downloader."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import SECFilingDownloader


class TestSECFilingDownloader:
    """Tests for the SECFilingDownloader class."""

    @pytest.fixture
    def mock_edgar_company(self):
        """Create a mock Company object."""
        mock_company = MagicMock()
        mock_company.name = "Test Company Inc."
        mock_company.cik = "1234567"

        # Mock filing
        mock_filing = MagicMock()
        mock_filing.filing_date = "2025-01-15"
        mock_filing.accession_number = "0001234567-25-000001"
        mock_filing.text.return_value = "Sample 10-K filing content"

        # Mock filings collection
        mock_filings = MagicMock()
        mock_filings.head.return_value = [mock_filing]
        mock_company.get_filings.return_value = mock_filings

        return mock_company

    @patch("src.data.downloader.set_identity")
    def test_downloader_initialization(self, mock_set_identity, temp_data_dir):
        """Test downloader initializes correctly."""
        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))

        assert downloader.output_dir == str(temp_data_dir["raw"])
        mock_set_identity.assert_called_once()

    @patch("src.data.downloader.set_identity")
    def test_downloader_creates_output_dir(self, mock_set_identity, tmp_path):
        """Test downloader creates output directory if it doesn't exist."""
        new_dir = tmp_path / "new_output"
        downloader = SECFilingDownloader(output_dir=str(new_dir))

        assert new_dir.exists()

    @patch("src.data.downloader.Company")
    @patch("src.data.downloader.set_identity")
    def test_download_10k_filings_success(
        self, mock_set_identity, mock_company_class, mock_edgar_company, temp_data_dir
    ):
        """Test successful download of 10-K filings."""
        mock_company_class.return_value = mock_edgar_company

        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))
        results = downloader.download_10k_filings(tickers=["TEST"], num_filings=1)

        assert "TEST" in results
        assert len(results["TEST"]) == 1
        assert results["TEST"][0].endswith(".txt")

    @patch("src.data.downloader.Company")
    @patch("src.data.downloader.set_identity")
    def test_download_creates_metadata_file(
        self, mock_set_identity, mock_company_class, mock_edgar_company, temp_data_dir
    ):
        """Test that download creates metadata JSON file."""
        mock_company_class.return_value = mock_edgar_company

        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))
        results = downloader.download_10k_filings(tickers=["TEST"], num_filings=1)

        # Check metadata file exists
        ticker_dir = temp_data_dir["raw"] / "TEST"
        meta_files = list(ticker_dir.glob("*_meta.json"))
        assert len(meta_files) == 1

        # Check metadata content
        with open(meta_files[0]) as f:
            metadata = json.load(f)

        assert metadata["ticker"] == "TEST"
        assert metadata["company_name"] == "Test Company Inc."
        assert metadata["form_type"] == "10-K"

    @patch("src.data.downloader.Company")
    @patch("src.data.downloader.set_identity")
    def test_download_handles_company_error(
        self, mock_set_identity, mock_company_class, temp_data_dir
    ):
        """Test that download handles errors gracefully."""
        mock_company_class.side_effect = Exception("Company not found")

        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))
        results = downloader.download_10k_filings(tickers=["INVALID"], num_filings=1)

        assert "INVALID" in results
        assert results["INVALID"] == []

    @patch("src.data.downloader.Company")
    @patch("src.data.downloader.set_identity")
    def test_download_multiple_tickers(
        self, mock_set_identity, mock_company_class, mock_edgar_company, temp_data_dir
    ):
        """Test downloading for multiple tickers."""
        mock_company_class.return_value = mock_edgar_company

        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))
        results = downloader.download_10k_filings(
            tickers=["AAPL", "MSFT", "GOOGL"], num_filings=1
        )

        assert len(results) == 3
        assert all(ticker in results for ticker in ["AAPL", "MSFT", "GOOGL"])

    @patch("src.data.downloader.set_identity")
    def test_get_download_summary(self, mock_set_identity, temp_data_dir, capsys):
        """Test download summary output."""
        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))

        # Create a mock file for summary
        ticker_dir = temp_data_dir["raw"] / "TEST"
        ticker_dir.mkdir()
        test_file = ticker_dir / "TEST_10K_2025-01-15.txt"
        test_file.write_text("Sample content")

        results = {"TEST": [str(test_file)], "EMPTY": []}
        downloader.get_download_summary(results)

        captured = capsys.readouterr()
        assert "DOWNLOAD SUMMARY" in captured.out
        assert "TEST:" in captured.out
        assert "EMPTY: No filings downloaded" in captured.out

    @patch("src.data.downloader.Company")
    @patch("src.data.downloader.set_identity")
    def test_download_saves_text_content(
        self, mock_set_identity, mock_company_class, mock_edgar_company, temp_data_dir
    ):
        """Test that filing text content is saved correctly."""
        mock_company_class.return_value = mock_edgar_company

        downloader = SECFilingDownloader(output_dir=str(temp_data_dir["raw"]))
        results = downloader.download_10k_filings(tickers=["TEST"], num_filings=1)

        # Read saved file
        saved_file = results["TEST"][0]
        with open(saved_file, encoding="utf-8") as f:
            content = f.read()

        assert content == "Sample 10-K filing content"
