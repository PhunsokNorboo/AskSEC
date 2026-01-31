"""Tests for Configuration module."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Settings


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_initialization(self):
        """Test settings initializes with default values."""
        settings = Settings()

        assert settings.PROJECT_ROOT.exists()
        assert settings.DATA_DIR.name == "data"
        assert settings.RAW_DATA_DIR.name == "raw"
        assert settings.PROCESSED_DATA_DIR.name == "processed"
        assert settings.CHROMA_DB_DIR.name == "chroma_db"

    def test_default_ollama_settings(self):
        """Test default Ollama configuration."""
        settings = Settings()

        assert settings.OLLAMA_MODEL == "llama3.2"
        assert settings.OLLAMA_BASE_URL == "http://localhost:11434"

    def test_default_embedding_settings(self):
        """Test default embedding configuration."""
        settings = Settings()

        assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_default_rag_settings(self):
        """Test default RAG configuration."""
        settings = Settings()

        assert settings.CHUNK_SIZE == 1500  # Larger for better financial context
        assert settings.CHUNK_OVERLAP == 300
        assert settings.RETRIEVAL_K == 6

    def test_default_chroma_settings(self):
        """Test default ChromaDB configuration."""
        settings = Settings()

        assert settings.CHROMA_COLLECTION_NAME == "sec_filings"

    @patch.dict(os.environ, {"OLLAMA_MODEL": "llama3.3"})
    def test_settings_reads_env_vars(self):
        """Test settings reads from environment variables."""
        settings = Settings()

        assert settings.OLLAMA_MODEL == "llama3.3"

    @patch.dict(os.environ, {"CHUNK_SIZE": "2000", "CHUNK_OVERLAP": "400"})
    def test_settings_converts_numeric_env_vars(self):
        """Test settings converts numeric environment variables."""
        settings = Settings()

        assert settings.CHUNK_SIZE == 2000
        assert settings.CHUNK_OVERLAP == 400
        assert isinstance(settings.CHUNK_SIZE, int)
        assert isinstance(settings.CHUNK_OVERLAP, int)

    def test_ensure_directories_creates_dirs(self, tmp_path):
        """Test ensure_directories creates required directories."""
        with patch.object(Settings, "__init__", lambda self: None):
            settings = Settings()
            settings.RAW_DATA_DIR = tmp_path / "raw"
            settings.PROCESSED_DATA_DIR = tmp_path / "processed"
            settings.CHROMA_DB_DIR = tmp_path / "chroma"

            # Directories should not exist yet
            assert not settings.RAW_DATA_DIR.exists()

            settings.ensure_directories()

            # Now they should exist
            assert settings.RAW_DATA_DIR.exists()
            assert settings.PROCESSED_DATA_DIR.exists()
            assert settings.CHROMA_DB_DIR.exists()

    def test_validate_returns_false_with_default_identity(self):
        """Test validate returns False when EDGAR_IDENTITY is not configured."""
        with patch.dict(
            os.environ, {"EDGAR_IDENTITY": "Your Name your.email@example.com"}
        ):
            settings = Settings()
            result = settings.validate()

            assert result is False

    @patch.dict(os.environ, {"EDGAR_IDENTITY": "John Doe john@example.com"})
    def test_validate_returns_true_with_configured_identity(self):
        """Test validate returns True when EDGAR_IDENTITY is configured."""
        settings = Settings()
        result = settings.validate()

        assert result is True

    def test_project_root_contains_expected_files(self):
        """Test project root is correctly identified."""
        settings = Settings()

        # Project root should contain src directory
        src_dir = settings.PROJECT_ROOT / "src"
        assert src_dir.exists() or (settings.PROJECT_ROOT / "tests").exists()

    @patch.dict(os.environ, {"RETRIEVAL_K": "10"})
    def test_retrieval_k_from_env(self):
        """Test RETRIEVAL_K can be configured via environment."""
        settings = Settings()

        assert settings.RETRIEVAL_K == 10

    @patch.dict(os.environ, {"EMBEDDING_MODEL": "paraphrase-MiniLM-L6-v2"})
    def test_custom_embedding_model(self):
        """Test custom embedding model can be configured."""
        settings = Settings()

        assert settings.EMBEDDING_MODEL == "paraphrase-MiniLM-L6-v2"
