"""
System & Information Routes
===========================
Health check, available languages, sample queries, and UI file serving.
"""

from pathlib import Path
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from core.config import (
    STATIC_DIR,
    DEVICE,
    GROQ_MODEL_NAME,
)
from core.tracing import tracing_enabled, project_name
from pipeline.router import LanguageRouter
from api.dependencies import get_router
from schemas.models import LanguageInfo
from schemas.responses import HealthResponse, AvailableLanguagesResponse

router = APIRouter()


@router.get("/", tags=["UI"])
@router.get("/ui", tags=["UI"])
def serve_ui():
    """Serves the interactive Multi-Language Voice & Text RAG web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "Voice RAG API is running. UI file not found."}


@router.get("/api/v1/languages", response_model=AvailableLanguagesResponse, tags=["Languages"])
def get_languages(router_instance: LanguageRouter = Depends(get_router)):
    """Returns list of supported and active languages in the system."""
    languages_info = router_instance.get_available_languages()
    return AvailableLanguagesResponse(
        languages=[LanguageInfo(**item) for item in languages_info],
        default_language="gu"
    )


@router.get("/api/info", tags=["Info"])
def api_info():
    """Returns general service info and API status."""
    return {
        "title": "Multi-Language Voice & Text RAG API",
        "version": "3.3.0",
        "status": "online",
        "supported_languages": ["gu", "hi"],
        "docs_url": "/docs",
        "langsmith_tracing": tracing_enabled(),
        "langsmith_project": project_name() if tracing_enabled() else None,
        "endpoints": {
            "languages": "GET /api/v1/languages",
            "text_query": "POST /api/v1/query/text",
            "voice_query": "POST /api/v1/query/voice",
            "streaming_query": "POST /api/v1/query/text/stream",
            "retrieve_only": "POST /api/v1/retrieve",
            "evaluate_only": "POST /api/v1/evaluate",
            "health": "GET /health"
        }
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(router_instance: LanguageRouter = Depends(get_router)):
    """Returns service health, loaded vector count per language, and database stats."""
    langs = router_instance.get_available_languages()
    total_vectors = sum(item["indexed_passages"] for item in langs)
    total_golden = sum(item["golden_dataset_records"] for item in langs)

    details = {
        item["code"]: {
            "name": item["name"],
            "indexed_passages": item["indexed_passages"],
            "golden_records": item["golden_dataset_records"]
        }
        for item in langs
    }

    return HealthResponse(
        status="healthy",
        device=str(DEVICE),
        loaded_languages=[item["code"] for item in langs],
        indexed_passages=total_vectors,
        golden_dataset_records=total_golden,
        embedding_model="BAAI/bge-m3",
        llm_model=f"{GROQ_MODEL_NAME} (Groq)",
        language_details=details
    )


@router.get("/api/v1/sample-queries", tags=["Queries"])
def get_sample_queries(
    lang: str = Query(default="gu", description="Language code ('gu' / 'hi')"),
    count: int = Query(default=20, ge=1, le=100),
    router_instance: LanguageRouter = Depends(get_router)
):
    """
    Returns N randomly sampled evaluation queries from the Golden Dataset for the given language.
    """
    pipeline = router_instance.get_pipeline(lang)
    samples = pipeline.get_sample_queries(count=count)
    return {
        "status": "success",
        "language": pipeline.lang_code,
        "count": len(samples),
        "queries": samples
    }
