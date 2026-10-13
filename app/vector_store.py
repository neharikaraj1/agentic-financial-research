"""
Vector store module using ChromaDB + LangChain.
Handles document ingestion, chunking, embedding, and similarity search.
"""

import os
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.config import get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB vector store:
    - Persist embeddings to disk
    - Add / delete documents
    - Similarity search with metadata filtering
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._ensure_directories()

        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        self.vector_store = self._load_or_create_store()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        Path(self.settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.settings.upload_dir).mkdir(parents=True, exist_ok=True)

    def _load_or_create_store(self) -> Chroma:
        logger.info(
            "Loading ChromaDB from '{}'", self.settings.chroma_persist_directory
        )
        return Chroma(
            collection_name=self.settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_persist_directory,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        source_name: str,
    ) -> List[str]:
        """
        Split `documents` into chunks and embed them into the vector store.
        Returns list of added document IDs.
        """
        logger.info("Splitting {} document(s) from '{}'", len(documents), source_name)
        chunks = self.text_splitter.split_documents(documents)

        # Inject source metadata into every chunk
        for chunk in chunks:
            chunk.metadata.setdefault("source", source_name)

        ids = self.vector_store.add_documents(chunks)
        logger.info("Added {} chunks to ChromaDB", len(chunks))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """Return the top-k most relevant document chunks for `query`."""
        logger.debug("Running similarity search: k={}, filter={}", k, filter)
        results = self.vector_store.similarity_search(query, k=k, filter=filter)
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple[Document, float]]:
        """Return chunks together with their cosine similarity scores."""
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)

    def delete_document(self, source_name: str) -> None:
        """Remove all chunks that belong to a given source document."""
        collection = self.vector_store._collection  # type: ignore[attr-defined]
        results = collection.get(where={"source": source_name})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted {} chunks for source '{}'", len(ids_to_delete), source_name
            )

    def list_sources(self) -> List[str]:
        """Return a deduplicated list of all ingested source names."""
        collection = self.vector_store._collection  # type: ignore[attr-defined]
        results = collection.get(include=["metadatas"])
        sources = {
            m.get("source", "unknown")
            for m in results.get("metadatas", [])
            if m
        }
        return sorted(sources)

    def get_retriever(self, k: int = 5):
        """Return a LangChain-compatible retriever for use in chains."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------

_vs_manager: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _vs_manager
    if _vs_manager is None:
        _vs_manager = VectorStoreManager()
    return _vs_manager
