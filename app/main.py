"""
FastAPI Application Entry Point.

Mounts all routers and configures:
  - CORS middleware
  - Lifespan events (startup checks)
  - OpenAPI docs at /docs
  - Health check at /health
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.routers import documents, query, evaluation
from app.schemas import HealthResponse


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before serving requests."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("🚀 Agentic Financial Research System starting up")
    logger.info("=" * 60)

    # Ensure data directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Warn if API key is not set
    if settings.openai_api_key.startswith("sk-placeholder"):
        logger.warning(
            "⚠️  OPENAI_API_KEY is not set. Set it in your .env file before running queries."
        )
    else:
        logger.info("✅ OpenAI API key detected")

    logger.info("✅ ChromaDB directory: {}", settings.chroma_persist_directory)
    logger.info("✅ Upload directory: {}", settings.upload_dir)
    logger.info("✅ Model: {}", settings.openai_model)
    logger.info("System ready — navigate to http://localhost:{}/docs", settings.app_port)

    yield  # Application runs here

    logger.info("Shutting down gracefully...")


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Agentic AI Financial Research & Document Analysis System",
        description="""
## Overview
A **LangGraph-based Agentic AI** system that decomposes complex financial queries
into multi-step tasks and dynamically routes them across:

- **Retrieval** (ChromaDB RAG)
- **Calculation** (financial ratio tools)
- **Analysis** (live stock data via yfinance)

## Workflow
1. `query_planner` — Breaks query into a structured research plan
2. `retriever` — Fetches relevant passages from ingested financial documents
3. `tool_executor` — Calls calculators, stock APIs, etc.
4. `reasoner` — Synthesises evidence into a draft answer
5. `validator` — Validates factual grounding
6. `response_gen` — Generates the final polished response

## Evaluation
RAGAs metrics (faithfulness, answer relevancy, context precision/recall)
are available via the `/api/evaluate` endpoint.
        """,
        version="1.0.0",
        contact={
            "name": "Agentic Financial Research System",
            "url": "https://github.com/your-username/agentic-financial-research",
        },
        license_info={"name": "MIT"},
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(evaluation.router)

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """System health check — verify all services are reachable."""
        services = {"fastapi": "ok", "chromadb": "unknown", "openai": "unknown"}

        try:
            from app.vector_store import get_vector_store
            vs = get_vector_store()
            vs.list_sources()  # lightweight ping
            services["chromadb"] = "ok"
        except Exception as e:
            services["chromadb"] = f"error: {str(e)}"

        if not settings.openai_api_key.startswith("sk-placeholder"):
            services["openai"] = "configured"
        else:
            services["openai"] = "not configured"

        return HealthResponse(status="ok", version="1.0.0", services=services)

    @app.get("/", tags=["System"])
    async def root():
        return {
            "message": "Agentic Financial Research & Document Analysis System",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# ---------------------------------------------------------------------------
# App instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
