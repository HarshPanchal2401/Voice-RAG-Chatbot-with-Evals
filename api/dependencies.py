"""
FastAPI Dependencies
====================
Provides dependency-injected access to application state (Router, VoiceService).
Gracefully self-heals if app state is accessed outside lifespan.
"""

from fastapi import Request
from pipeline.router import LanguageRouter
from services.voice_service import VoiceService


def get_router(request: Request) -> LanguageRouter:
    if not hasattr(request.app.state, "router") or request.app.state.router is None:
        request.app.state.router = LanguageRouter(verbose=False)
    return request.app.state.router


def get_voice_service(request: Request) -> VoiceService:
    if not hasattr(request.app.state, "voice_service") or request.app.state.voice_service is None:
        router = get_router(request)
        request.app.state.voice_service = VoiceService(groq_client=router.groq_client)
    return request.app.state.voice_service
