"""
System & Information Routes
===========================
Health check, available languages, sample queries, API info and UI file serving.
"""

import os

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute, APIWebSocketRoute

from core.config import STATIC_DIR, DEVICE, GROQ_MODEL_NAME
from core.security import PROTECTED, auth_enabled
from core.tracing import tracing_enabled, project_name
from api.dependencies import get_router
from api.helpers import sanitize_sample_queries
from schemas.models import LanguageInfo
from schemas.requests import normalize_request_language
from schemas.responses import HealthResponse, AvailableLanguagesResponse, SampleQueriesResponse

try:  # names owned by the pipeline helper in core/config.py
    from core.config import RERANKER_BACKEND as _CONFIG_RERANKER_BACKEND  # type: ignore
except ImportError:
    _CONFIG_RERANKER_BACKEND = None
try:
    from core.config import BGE_MODEL_NAME  # type: ignore
except ImportError:
    BGE_MODEL_NAME = "BAAI/bge-m3"

API_VERSION = "3.4.0"

router = APIRouter()


def reranker_backend() -> str:
    return _CONFIG_RERANKER_BACKEND or (os.environ.get("RAG_RERANKER_BACKEND") or "llm").strip().lower()


@router.get("/", tags=["UI"], include_in_schema=False)
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
                "Expires": "0",
            },
        )
    return {"message": "Voice RAG API is running. UI file not found."}


@router.get("/api/v1/languages", response_model=AvailableLanguagesResponse, tags=["Languages"], dependencies=PROTECTED)
def get_languages(router_instance=Depends(get_router)):
    """Returns list of supported and active languages in the system."""
    languages_info = router_instance.get_available_languages()
    return AvailableLanguagesResponse(
        languages=[LanguageInfo(**item) for item in languages_info],
        default_language="gu",
    )


@router.get("/api/info", tags=["Info"], dependencies=PROTECTED)
def api_info(request: Request):
    """General service info and the endpoints actually registered on this server."""
    endpoints = []
    for route in request.app.routes:
        if isinstance(route, APIRoute) and route.include_in_schema:
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                endpoints.append(f"{method} {route.path}")
        elif isinstance(route, APIWebSocketRoute):
            endpoints.append(f"WS {route.path}")
    return {
        "title": request.app.title,
        "version": request.app.version,
        "status": "online",
        "supported_languages": ["gu", "hi"],
        "answer_languages": ["gu", "hi", "en"],
        "docs_url": "/docs",
        "auth_required": auth_enabled(),
        "langsmith_tracing": tracing_enabled(),
        "langsmith_project": project_name() if tracing_enabled() else None,
        "endpoints": sorted(set(endpoints)),
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(router_instance=Depends(get_router)):
    """Service health, loaded vector count per language and model configuration (public, no secrets)."""
    langs = router_instance.get_available_languages()
    details = {
        item["code"]: {
            "name": item["name"],
            "indexed_passages": item["indexed_passages"],
            "golden_records": item["golden_dataset_records"],
        }
        for item in langs
    }
    return HealthResponse(
        status="healthy" if langs else "degraded",
        device=str(DEVICE),
        loaded_languages=[item["code"] for item in langs],
        indexed_passages=sum(item["indexed_passages"] for item in langs),
        golden_dataset_records=sum(item["golden_dataset_records"] for item in langs),
        embedding_model=BGE_MODEL_NAME,
        llm_model=GROQ_MODEL_NAME,
        reranker_backend=reranker_backend(),
        auth_required=auth_enabled(),
        tracing_enabled=tracing_enabled(),
        language_details=details,
    )


@router.get("/api/v1/sample-queries", response_model=SampleQueriesResponse, tags=["Queries"], dependencies=PROTECTED)
def get_sample_queries(
    lang: str = Query(default="gu", description="Language code ('gu' / 'hi')"),
    count: int = Query(default=20, ge=1, le=100),
    router_instance=Depends(get_router),
):
    """N random golden-dataset questions for the language (question, query_type, query_id only)."""
    try:
        code = normalize_request_language(lang, allow_auto=False)
    except ValueError:
        code = "gu"
    pipeline = router_instance.get_pipeline(code)
    samples = sanitize_sample_queries(pipeline.get_sample_queries(count=count))
    return SampleQueriesResponse(
        status="success",
        language=pipeline.lang_code,
        count=len(samples),
        queries=samples,
    )
