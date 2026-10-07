"""
Retrieval Routes
================
Search-only endpoint for testing and integrating the hybrid retrieval and sorting engine.
"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from core.security import PROTECTED_LIMITED
from api.dependencies import get_router
from api.helpers import build_retrieval_timings, format_sources
from schemas.requests import RetrieveRequest
from schemas.responses import RetrievalResponse

router = APIRouter()


@router.post("/api/v1/retrieve", response_model=RetrievalResponse, tags=["Retrieval"], dependencies=PROTECTED_LIMITED)
async def retrieve_passages(
    payload: RetrieveRequest,
    router_instance=Depends(get_router),
):
    """
    Search-only: ranked passages for the target language with RRF scores, dense/sparse ranks and
    SQLite metadata. Sorting: 'rrf' | 'dense' | 'sparse' | 'rerank'; optional re-ranking.
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
        sources=format_sources(retrieval_result.get("documents") or [], lang=target_lang),
        retrieval_timings=build_retrieval_timings(retrieval_result.get("timings")),
    )
