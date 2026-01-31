"""Shared test fixtures for AskSEC tests."""
import pytest
from pathlib import Path


@pytest.fixture
def sample_10k_text():
    """Sample 10-K filing text for testing."""
    return """
    UNITED STATES
    SECURITIES AND EXCHANGE COMMISSION
    Washington, D.C. 20549
    FORM 10-K

    ITEM 1. BUSINESS

    We are a technology company that develops innovative products.
    Our main business segments include hardware, software, and services.
    We operate globally with presence in Americas, Europe, and Asia.

    ITEM 1A. RISK FACTORS

    Investing in our securities involves risks. The following risk factors
    should be considered carefully:

    - Competition from other technology companies may reduce our market share.
    - Economic downturns could reduce consumer spending on our products.
    - Supply chain disruptions may affect our ability to deliver products.
    - Regulatory changes could increase our compliance costs.

    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

    Our revenue increased 15% year over year to $100 billion.
    Operating expenses grew by 10% primarily due to R&D investments.
    Net income was $25 billion, representing a 25% profit margin.

    ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA

    Consolidated Balance Sheet
    Total Assets: $350 billion
    Total Liabilities: $200 billion
    Stockholders Equity: $150 billion
    """


@pytest.fixture
def sample_metadata():
    """Sample filing metadata for testing."""
    return {
        "ticker": "TEST",
        "company_name": "Test Company Inc.",
        "filing_date": "2025-01-15",
        "accession_number": "0001234567-25-000001",
        "form_type": "10-K",
        "cik": "1234567"
    }


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    chroma_dir = tmp_path / "chroma_db"

    raw_dir.mkdir()
    processed_dir.mkdir()
    chroma_dir.mkdir()

    return {
        "root": tmp_path,
        "raw": raw_dir,
        "processed": processed_dir,
        "chroma": chroma_dir
    }
