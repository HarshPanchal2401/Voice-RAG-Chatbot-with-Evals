"""
Retrieval Routes
================
Search-only endpoint for testing and integrating the hybrid retrieval and sorting engine.
"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from pipeline.router import LanguageRouter
from api.dependencies import get_router
from api.helpers import format_sources
from schemas.requests import RetrieveRequest
from schemas.responses import RetrievalResponse
from schemas.models import RetrievalTimings

router = APIRouter()


@router.post("/api/v1/retrieve", response_model=RetrievalResponse, tags=["Retrieval"])
async def retrieve_passages(
    payload: RetrieveRequest,
    router_instance: LanguageRouter = Depends(get_router)
):
    """
    Search-Only Endpoint:
    Returns ranked passages for target language with RRF scores, dense/sparse ranks, and SQLite metadata.
    Supports multi-strategy sorting ('rrf', 'dense', 'sparse', 'rerank') and secondary listwise re-ranking.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    target_lang = router_instance.resolve_language(
        query=query,
        requested_lang=payload.language,
        auto_detect=payload.auto_detect_language,
    )
    pipeline = router_instance.get_pipeline(target_lang)

    retrieval_result = await run_in_threadpool(
        pipeline.hybrid_retrieve,
        query=query,
        final_k=payload.top_k,
        use_reranker=payload.use_reranker,
        sort_by=payload.sort_by,
    )

    return RetrievalResponse(
        status="success",
        query=query,
        language=target_lang,
        top_k=payload.top_k,
        sources=format_sources(retrieval_result["documents"], lang=target_lang),
        retrieval_timings=RetrievalTimings(**retrieval_result["timings"])
    )
