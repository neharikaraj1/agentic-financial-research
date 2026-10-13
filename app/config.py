"""
Application configuration using Pydantic Settings.
Loads from environment variables / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "financial_documents"

    # FastAPI
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False
    secret_key: str = "change-me"

    # Document Processing
    upload_dir: str = "./data/uploads"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Agent
    max_iterations: int = 10
    verbose: bool = False

    # RAGAs
    ragas_eval_enabled: bool = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
