"""
Evaluation Routes
=================
Standalone DeepEval evaluation endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from core.security import PROTECTED_LIMITED
from api.dependencies import get_router
from api.helpers import format_evaluation
from schemas.requests import EvaluateRequest
from schemas.models import EvaluationResult

router = APIRouter()


@router.post("/api/v1/evaluate", response_model=EvaluationResult, tags=["Evaluation"], dependencies=PROTECTED_LIMITED)
async def evaluate_standalone(
    payload: EvaluateRequest,
    router_instance=Depends(get_router),
):
    """
    Standalone DeepEval evaluation for the target language:
    faithfulness, answer relevancy, contextual recall and answer correctness.
    """
    target_lang = router_instance.resolve_language(
        query=payload.question,
        requested_lang=payload.language,
        auto_detect=True,
    )
    pipeline = router_instance.get_pipeline(target_lang)

    if not pipeline.evaluator:
        raise HTTPException(status_code=503, detail=f"DeepEval evaluator not available for '{target_lang}'.")

    eval_result = await run_in_threadpool(
        pipeline.evaluator.evaluate,
        question=payload.question,
        generated_answer=payload.generated_answer,
        retrieved_contexts=payload.retrieved_contexts,
        query_id=payload.query_id,
    )
    return format_evaluation(eval_result)
