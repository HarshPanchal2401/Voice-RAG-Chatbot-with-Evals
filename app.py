"""
FastAPI Server for Multi-Language Voice & Text RAG System (v3)
===============================================================
Modular routing, in-memory model loading in the lifespan, speech-to-text, hybrid retrieval,
sentence-level streaming TTS, LangSmith tracing and DeepEval evaluation.

Security (see core/security.py): `RAG_API_KEY` (X-API-Key header / `api_key` WS query param),
`RAG_RATE_LIMIT_PER_MIN`, `RAG_CORS_ORIGINS`, `RAG_MAX_UPLOAD_MB`, `RAG_WS_MAX_SECONDS`.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from core.config import STATIC_DIR  # noqa: E402
from core.security import UploadSizeLimitMiddleware, cors_origins, log_security_config  # noqa: E402
from core.tracing import tracing_enabled, project_name, flush as flush_traces  # noqa: E402
from pipeline.router import LanguageRouter  # noqa: E402
from services.voice_service import VoiceService  # noqa: E402
from api import (  # noqa: E402
    system_router,
    query_router,
    stream_router,
    retrieval_router,
    evaluation_router,
    feedback_router,
)
from api.routes_system import API_VERSION  # noqa: E402


# ============================================================
# Lifespan (models are loaded once, in memory)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    created_here = False
    # Tests (and embedding apps) can inject `app.state.router` / `app.state.voice_service`
    # before startup; the heavy LanguageRouter is only built when nothing was injected.
    if getattr(app.state, "router", None) is None:
        print("=" * 60)
        print("🚀 [FastAPI Lifespan] Initializing Multi-Language RAG Pipeline Router...")
        print("=" * 60)

        t0 = time.time()
        router = LanguageRouter(verbose=True)
        app.state.router = router
        app.state.pipeline = router.get_pipeline("gu")  # backward compat
        if getattr(app.state, "voice_service", None) is None:
            app.state.voice_service = VoiceService(groq_client=router.groq_client)
        created_here = True

        print("=" * 60)
        print(f"✅ [FastAPI Lifespan] All Language Pipelines loaded in {time.time() - t0:.2f}s!")

    if tracing_enabled():
        print(f"🛰️ [LangSmith] Tracing ON -> project '{project_name()}'")
    else:
        print("ℹ️ [LangSmith] Tracing OFF (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
    log_security_config()
    print("=" * 60)

    yield

    print("🛑 [FastAPI Lifespan] Shutting down RAG services...")
    flush_traces()
    if created_here:
        voice_service = getattr(app.state, "voice_service", None)
        if voice_service is not None and hasattr(voice_service, "close"):
            voice_service.close()
        router = getattr(app.state, "router", None)
        if router is not None:
            router.close()


# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="Multi-Language Voice & Text RAG API",
    description=(
        "Low-latency Voice & Text RAG API for Gujarati and Hindi: speech-to-text, hybrid retrieval, "
        "history-aware answers, sentence-level streaming TTS, LangSmith tracing and DeepEval evaluation. "
        "When the server sets RAG_API_KEY, /api/* routes require the `X-API-Key` header."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# Order matters: the last middleware added is the outermost, so CORS headers are also
# attached to the 413 responses produced by the upload limiter.
app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    expose_headers=["Retry-After"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(system_router)
app.include_router(query_router)
app.include_router(stream_router)
app.include_router(retrieval_router)
app.include_router(evaluation_router)
app.include_router(feedback_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="localhost", port=8000, reload=False)
