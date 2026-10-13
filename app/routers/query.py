"""
FastAPI routers — Agent Query endpoints.

POST /api/query        - Run the full LangGraph agentic pipeline
POST /api/query/quick  - Quick RAG-only retrieval (no full agent loop)
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.agent import run_agent
from app.schemas import AgentResponse, QueryRequest, ToolOutput
from app.vector_store import get_vector_store

router = APIRouter(prefix="/api/query", tags=["Agent Query"])


@router.post(
    "",
    response_model=AgentResponse,
    summary="Run the full agentic financial research pipeline",
    description="""
Executes the complete LangGraph pipeline:
1. **Query Planner** – Decomposes the query into a research plan
2. **Retriever** – Fetches relevant document chunks via RAG
3. **Tool Executor** – Calls financial calculators / stock data APIs as needed
4. **Reasoner** – Synthesises evidence into a draft answer
5. **Validator** – Checks factual grounding
6. **Response Generator** – Produces the final polished answer
""",
)
async def run_query(request: QueryRequest):
    """Execute the full agentic pipeline for a financial research query."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty.",
        )

    logger.info("Received query: '{}'", request.query)

    try:
        result = run_agent(request.query)
        return AgentResponse(
            query=request.query,
            plan=result.get("plan", []),
            final_answer=result.get("final_answer", ""),
            sources=result.get("sources", []),
            tool_outputs=[
                ToolOutput(
                    tool=t["tool"],
                    input=t["input"],
                    output=str(t["output"]),
                )
                for t in result.get("tool_outputs", [])
            ],
            validation_passed=result.get("validation_passed", True),
            validation_notes=result.get("validation_notes", ""),
            iterations=result.get("iteration", 1),
        )
    except Exception as e:
        logger.error("Agent execution failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}",
        )


@router.post(
    "/quick",
    summary="Quick RAG retrieval (no full agent loop)",
    description="Directly retrieves relevant document chunks for a query without running the full agent.",
)
async def quick_retrieval(request: QueryRequest):
    """Return top-k relevant chunks for a query (fast, no LLM reasoning)."""
    try:
        vs = get_vector_store()
        results = vs.similarity_search_with_score(request.query, k=5)
        chunks = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "score": round(score, 4),
            }
            for doc, score in results
        ]
        return {"query": request.query, "chunks": chunks, "total": len(chunks)}
    except Exception as e:
        logger.error("Quick retrieval failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
