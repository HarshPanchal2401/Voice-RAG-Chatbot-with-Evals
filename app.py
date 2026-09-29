"""
FastAPI Server for Multi-Language Voice & Text RAG System (v3)
===============================================================
Production-grade server with modular routing, in-memory Lifespan caching,
speech-to-text, hybrid retrieval, deterministic sorting, and LangSmith tracing.
"""

import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from core.config import STATIC_DIR
from core.tracing import tracing_enabled, project_name, flush as flush_traces
from pipeline.router import LanguageRouter
from services.voice_service import VoiceService
from api import (
    system_router,
    query_router,
    stream_router,
    retrieval_router,
    evaluation_router,
)

load_dotenv()


# ============================================================
# Lifespan Management (In-Memory Model Loading)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "router") or app.state.router is None:
        print("=" * 60)
        print("🚀 [FastAPI Lifespan] Initializing Multi-Language RAG Pipeline Router...")
        print("=" * 60)

        t0 = time.time()
        router = LanguageRouter(verbose=True)
        voice_service = VoiceService(groq_client=router.groq_client)

        app.state.router = router
        app.state.pipeline = router.get_pipeline("gu")  # backward compat
        app.state.voice_service = voice_service

        print("=" * 60)
        print(f"✅ [FastAPI Lifespan] All Language Pipelines loaded in {time.time() - t0:.2f}s!")
    else:
        router = app.state.router

    if tracing_enabled():
        print(f"🛰️ [LangSmith] Tracing ON -> project '{project_name()}'")
    else:
        print("ℹ️ [LangSmith] Tracing OFF (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
    print("=" * 60)

    yield

    print("🛑 [FastAPI Lifespan] Shutting down RAG services...")
    flush_traces()
    if hasattr(app.state, "router") and app.state.router:
        app.state.router.close()


# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="Multi-Language Voice & Text RAG API",
    description="High-performance, low-latency Voice & Text RAG API for Gujarati and Hindi with Speech-to-Text, latency profiling, LangSmith tracing and DeepEval evaluation.",
    version="3.3.0",
    lifespan=lifespan
)

# Enable CORS for web and mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include Modular API Routers
app.include_router(system_router)
app.include_router(query_router)
app.include_router(stream_router)
app.include_router(retrieval_router)
app.include_router(evaluation_router)


# Backward Compatibility Dependencies
def get_router() -> LanguageRouter:
    return app.state.router


def get_voice_service() -> VoiceService:
    return app.state.voice_service


# ============================================================
# Server Runner
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=False)
