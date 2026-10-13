"""
RAGAs evaluation module.
Computes retrieval-augmented generation quality metrics:
  - Faithfulness       : Is the answer grounded in the context?
  - Answer Relevancy   : Is the answer relevant to the question?
  - Context Precision  : Are the retrieved chunks relevant?
  - Context Recall     : Are all needed facts retrieved?
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import get_settings


def evaluate_rag(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
) -> Dict[str, Optional[float]]:
    """
    Run RAGAs evaluation on a list of QA samples.

    Args:
        questions:     List of input questions.
        answers:       List of generated answers (one per question).
        contexts:      List of context chunk lists (one list per question).
        ground_truths: Optional list of reference answers.

    Returns:
        Dict of metric names to average scores.
    """
    settings = get_settings()
    if not settings.ragas_eval_enabled:
        logger.info("RAGAs evaluation is disabled via config.")
        return {}

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        import openai

        openai.api_key = settings.openai_api_key

        data: Dict[str, List[Any]] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        metrics_to_run = [faithfulness, answer_relevancy, context_precision]

        if ground_truths and len(ground_truths) == len(questions):
            data["ground_truth"] = ground_truths
            metrics_to_run.append(context_recall)

        dataset = Dataset.from_dict(data)
        logger.info("Running RAGAs evaluation on {} samples", len(questions))

        result = evaluate(dataset, metrics=metrics_to_run)
        result_dict = result.to_pandas().mean(numeric_only=True).to_dict()

        logger.info("RAGAs results: {}", result_dict)
        return {
            "faithfulness": result_dict.get("faithfulness"),
            "answer_relevancy": result_dict.get("answer_relevancy"),
            "context_precision": result_dict.get("context_precision"),
            "context_recall": result_dict.get("context_recall"),
        }

    except ImportError:
        logger.warning("RAGAs not installed. Install with: pip install ragas")
        return {"error": "ragas not installed"}
    except Exception as e:
        logger.error("RAGAs evaluation failed: {}", e)
        return {"error": str(e)}
