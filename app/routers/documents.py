"""
FastAPI routers — Document Ingestion endpoints.

POST /api/documents/upload     - Upload a file (PDF, DOCX, TXT, XLSX)
POST /api/documents/text       - Ingest raw text
GET  /api/documents/           - List all ingested document sources
DELETE /api/documents/{source} - Remove a document from the knowledge base
"""

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.config import get_settings
from app.ingestion import ingest_file, ingest_text, SUPPORTED_EXTENSIONS
from app.schemas import (
    DeleteDocumentRequest,
    DocumentListResponse,
    IngestResponse,
    IngestTextRequest,
)
from app.vector_store import get_vector_store

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a financial document",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a financial document (PDF, DOCX, TXT, XLSX) to be chunked,
    embedded, and stored in ChromaDB for RAG retrieval.
    """
    settings = get_settings()
    ext = Path(file.filename or "").suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Save to upload directory
    upload_path = Path(settings.upload_dir) / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved uploaded file: {}", upload_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Ingest into vector store
    try:
        result = ingest_file(str(upload_path))
        return IngestResponse(
            source=result["source"],
            pages_loaded=result.get("pages_loaded"),
            chunks_stored=result["chunks_stored"],
            message=f"Successfully ingested '{file.filename}' — {result['chunks_stored']} chunks stored.",
        )
    except Exception as e:
        logger.error("Ingestion failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post(
    "/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text content",
)
async def ingest_text_endpoint(request: IngestTextRequest):
    """Ingest raw text directly into the knowledge base."""
    try:
        result = ingest_text(request.text, source_name=request.source_name)
        return IngestResponse(
            source=result["source"],
            chunks_stored=result["chunks_stored"],
            message=f"Ingested text as '{request.source_name}' — {result['chunks_stored']} chunks stored.",
        )
    except Exception as e:
        logger.error("Text ingestion failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all ingested document sources",
)
async def list_documents():
    """Return all document source names currently in the knowledge base."""
    try:
        vs = get_vector_store()
        sources = vs.list_sources()
        return DocumentListResponse(sources=sources, total=len(sources))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete(
    "/{source_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document from the knowledge base",
)
async def delete_document(source_name: str):
    """Remove all chunks for the given document source from ChromaDB."""
    try:
        vs = get_vector_store()
        vs.delete_document(source_name)
        return {"message": f"Document '{source_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
