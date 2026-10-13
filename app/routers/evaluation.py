"""
FastAPI routers — RAGAs Evaluation endpoints.

POST /api/evaluate   - Run RAGAs metrics on provided QA samples
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.evaluation import evaluate_rag
from app.schemas import EvaluationRequest, EvaluationResponse, EvaluationMetrics

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])


@router.post(
    "",
    response_model=EvaluationResponse,
    summary="Evaluate RAG quality using RAGAs metrics",
    description="""
Runs RAGAs evaluation metrics:
- **Faithfulness**: Are answers grounded in retrieved context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: Are retrieved chunks relevant?
- **Context Recall**: Are all needed facts retrieved? (requires ground_truth)

Each sample requires: question, answer, contexts (list of retrieved chunks).
Optionally include ground_truth for context_recall metric.
""",
)
async def evaluate_endpoint(request: EvaluationRequest):
    """Compute RAGAs metrics for a batch of QA evaluation samples."""
    if not request.samples:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one evaluation sample is required.",
        )

    questions = [s.question for s in request.samples]
    answers = [s.answer for s in request.samples]
    contexts = [s.contexts for s in request.samples]
    ground_truths = [s.ground_truth for s in request.samples if s.ground_truth]

    logger.info("Running RAGAs evaluation on {} samples", len(questions))

    try:
        metrics = evaluate_rag(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if ground_truths else None,
        )

        return EvaluationResponse(
            metrics=EvaluationMetrics(
                faithfulness=metrics.get("faithfulness"),
                answer_relevancy=metrics.get("answer_relevancy"),
                context_recall=metrics.get("context_recall"),
                context_precision=metrics.get("context_precision"),
            ),
            num_samples=len(questions),
            message=f"Evaluated {len(questions)} samples successfully.",
        )
    except Exception as e:
        logger.error("Evaluation failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
