"""
Pydantic schemas for FastAPI request/response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------

class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Raw text content to ingest")
    source_name: str = Field(
        default="inline-text",
        description="Label for this document in the knowledge base",
    )


class IngestResponse(BaseModel):
    source: str
    pages_loaded: Optional[int] = None
    chunks_stored: int
    message: str = "Document successfully ingested"


class DeleteDocumentRequest(BaseModel):
    source_name: str = Field(..., description="Source name to remove from ChromaDB")


# ---------------------------------------------------------------------------
# Query / Agent
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Financial research question")


class ToolOutput(BaseModel):
    tool: str
    input: Dict[str, Any]
    output: str


class AgentResponse(BaseModel):
    query: str
    plan: List[str]
    final_answer: str
    sources: List[str]
    tool_outputs: List[ToolOutput]
    validation_passed: bool
    validation_notes: str
    iterations: int


# ---------------------------------------------------------------------------
# Document Listing
# ---------------------------------------------------------------------------

class DocumentListResponse(BaseModel):
    sources: List[str]
    total: int


# ---------------------------------------------------------------------------
# RAGAs Evaluation
# ---------------------------------------------------------------------------

class EvaluationSample(BaseModel):
    question: str = Field(..., description="The test question")
    answer: str = Field(..., description="The generated answer to evaluate")
    contexts: List[str] = Field(..., description="Retrieved document chunks used")
    ground_truth: Optional[str] = Field(
        default=None, description="Reference answer (for answer similarity)"
    )


class EvaluationRequest(BaseModel):
    samples: List[EvaluationSample]


class EvaluationMetrics(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None


class EvaluationResponse(BaseModel):
    metrics: EvaluationMetrics
    num_samples: int
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    services: Dict[str, str]
