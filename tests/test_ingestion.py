"""
Tests for document ingestion pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch


def test_ingest_text_success():
    """Ingesting raw text should return chunk metadata."""
    mock_ids = ["id1", "id2"]

    with patch("app.ingestion.get_vector_store") as mock_vs_factory:
        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = mock_ids
        mock_vs_factory.return_value = mock_vs

        from app.ingestion import ingest_text
        result = ingest_text("Apple reported revenue of $90B in Q4 2024.", "test-doc")

    assert result["source"] == "test-doc"
    assert result["chunks_stored"] == 2


def test_ingest_unsupported_format():
    """Unsupported file extensions should raise ValueError."""
    from app.ingestion import load_document
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("report.xyz")


def test_supported_extensions():
    from app.ingestion import SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
