"""
FastAPI Dependencies
====================
Dependency-injected access to application state (LanguageRouter, VoiceService).

The heavy objects are created once by the app lifespan. If they are missing (startup still
running, failed, or the app is mounted without its lifespan) the request gets a fast 503
instead of silently loading BGE-M3 + FAISS indexes inside a request (minutes, racy, OOM-prone).
"""

from typing import Any, Optional

from fastapi import HTTPException, Request


def router_from_state(app: Any) -> Optional[Any]:
    return getattr(app.state, "router", None)


def voice_service_from_state(app: Any) -> Optional[Any]:
    return getattr(app.state, "voice_service", None)


def get_router(request: Request):
    router = router_from_state(request.app)
    if router is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipelines are not loaded yet (server starting or failed to start). Retry shortly.",
            headers={"Retry-After": "30"},
        )
    return router


def get_voice_service(request: Request):
    voice_service = voice_service_from_state(request.app)
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Voice service is not available.", headers={"Retry-After": "30"})
    return voice_service


def get_optional_voice_service(request: Request):
    """Voice service or None (text endpoints only need it when `voice_reply` is requested)."""
    return voice_service_from_state(request.app)
