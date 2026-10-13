"""
Document ingestion pipeline.
Supports PDF, DOCX, TXT, and plain text ingestion into ChromaDB.
"""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from loguru import logger

from app.config import get_settings
from app.vector_store import get_vector_store


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".xlsx"}


def load_document(file_path: str) -> List[Document]:
    """Load a document from disk using the appropriate LangChain loader."""
    ext = Path(file_path).suffix.lower()
    logger.info("Loading '{}' (type: {})", file_path, ext)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in {".txt", ".md"}:
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".xlsx":
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def ingest_file(file_path: str) -> dict:
    """
    Full ingestion pipeline for a single file:
    1. Load → 2. Split → 3. Embed → 4. Store in ChromaDB
    """
    settings = get_settings()
    file_name = Path(file_path).name

    documents = load_document(file_path)
    vs = get_vector_store()
    ids = vs.add_documents(documents, source_name=file_name)

    return {
        "source": file_name,
        "pages_loaded": len(documents),
        "chunks_stored": len(ids),
        "chunk_ids": ids,
    }


def ingest_text(text: str, source_name: str = "inline-text") -> dict:
    """Ingest raw text directly (no file needed)."""
    documents = [Document(page_content=text, metadata={"source": source_name})]
    vs = get_vector_store()
    ids = vs.add_documents(documents, source_name=source_name)
    return {
        "source": source_name,
        "chunks_stored": len(ids),
    }


def ingest_directory(directory: str) -> List[dict]:
    """Ingest all supported documents found in a directory."""
    results = []
    for path in Path(directory).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                result = ingest_file(str(path))
                results.append(result)
            except Exception as exc:
                logger.error("Failed to ingest '{}': {}", path, exc)
                results.append({"source": str(path), "error": str(exc)})
    return results
