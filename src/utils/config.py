"""Configuration management for SEC Filing RAG Application."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.CHROMA_DB_DIR = self.DATA_DIR / "chroma_db"

        # SEC EDGAR identity (required by SEC)
        self.EDGAR_IDENTITY = os.getenv(
            "EDGAR_IDENTITY",
            "Your Name your.email@example.com"
        )

        # Ollama settings
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Embedding settings
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # RAG settings (larger chunks for better financial context)
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
        self.RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

        # ChromaDB settings
        self.CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sec_filings")

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Validate that required settings are configured."""
        logger = logging.getLogger("asksec.config")
        issues: list[str] = []

        if self.EDGAR_IDENTITY == "Your Name your.email@example.com":
            issues.append("EDGAR_IDENTITY not configured in .env file")

        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        return True


# Global settings instance
settings = Settings()
