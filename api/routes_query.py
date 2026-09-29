"""
Query Routes (Text & Voice)
===========================
Primary endpoints for synchronous Text & Voice RAG execution.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool

from core.tracing import traceable, current_run_id, add_run_metadata
from pipeline.router import LanguageRouter
from pipeline.single_pipeline import SingleLanguagePipeline
from services.voice_service import VoiceService
from api.dependencies import get_router, get_voice_service
from api.helpers import format_sources, format_evaluation, build_latency, run_evaluation
from schemas.requests import TextQueryRequest
from schemas.responses import RAGResponse

router = APIRouter()


def _drop_objects(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k not in ("pipeline", "voice_service")}


def _voice_inputs(inputs: dict) -> dict:
    audio = inputs.get("audio_bytes") or b""
    return {k: v for k, v in inputs.items() if k not in ("audio_bytes", "pipeline", "voice_service")} | {
        "audio_kb": round(len(audio) / 1024, 1)
    }


def _rag_outputs(outputs: any) -> any:
    if isinstance(outputs, dict) and "documents" in outputs:
        return {k: v for k, v in outputs.items() if k != "documents"} | {"num_sources": len(outputs["documents"])}
    return outputs


@traceable(run_type="chain", name="text_query", process_inputs=_drop_objects, process_outputs=_rag_outputs)
def handle_text_query(
    pipeline: SingleLanguagePipeline,
    query: str,
    lang: str,
    top_k: int,
    evaluate: bool,
    query_id: Optional[int],
    use_reranker: bool = False,
    sort_by: str = "rrf",
) -> dict:
    root_id = current_run_id()
    rag_result = pipeline.ask(query, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by)
    eval_result, eval_ms = (None, None)
    if evaluate and pipeline.evaluator:
        eval_result, eval_ms = run_evaluation(pipeline, query, rag_result, query_id, root_id)
    add_run_metadata(
        language=lang,
        mode="text",
        total_ms=round(rag_result["total_ms"], 2),
        eval_ms=eval_ms,
        reranked=use_reranker,
        sort_by=sort_by,
    )
    return {**rag_result, "evaluation": eval_result, "eval_ms": eval_ms, "trace_id": rag_result.get("trace_id")}


@traceable(run_type="chain", name="voice_query", process_inputs=_voice_inputs, process_outputs=_rag_outputs)
def handle_voice_query(
    pipeline: SingleLanguagePipeline,
    voice_service: VoiceService,
    audio_bytes: bytes,
    filename: str,
    lang: str,
    top_k: int,
    evaluate: bool,
    query_id: Optional[int],
    use_reranker: bool = False,
    sort_by: str = "rrf",
) -> dict:
    root_id = current_run_id()
    transcription, stt_ms = voice_service.transcribe_audio_bytes(
        audio_bytes=audio_bytes, filename=filename, language=lang
    )
    if not transcription:
        return {"transcription": "", "stt_ms": stt_ms}

    rag_result = pipeline.ask(transcription, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by)
    eval_result, eval_ms = (None, None)
    if evaluate and pipeline.evaluator:
        eval_result, eval_ms = run_evaluation(pipeline, transcription, rag_result, query_id, root_id)
    add_run_metadata(
        language=lang,
        mode="voice",
        stt_ms=round(stt_ms, 2),
        total_ms=round(stt_ms + rag_result["total_ms"], 2),
        eval_ms=eval_ms,
        reranked=use_reranker,
        sort_by=sort_by,
    )
    return {
        **rag_result,
        "transcription": transcription,
        "stt_ms": stt_ms,
        "evaluation": eval_result,
        "eval_ms": eval_ms,
    }


@router.post("/api/v1/query/text", response_model=RAGResponse, tags=["Query"])
@router.post("/api/v1/ask", response_model=RAGResponse, tags=["Query"])
async def query_by_text(
    payload: TextQueryRequest,
    router_instance: LanguageRouter = Depends(get_router)
):
    """
    Primary Text RAG Endpoint:
    1. Smart language routing (auto script detection if enabled).
    2. Hybrid retrieval (Dense FAISS + Sparse CSC + RRF + SQLite).
    3. Multi-strategy sorting & optional listwise re-ranking.
    4. Groq LLM answer generation in target language.
    5. Latency profiling & optional DeepEval scoring.
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

    result = await run_in_threadpool(
        handle_text_query,
        pipeline,
        query,
        target_lang,
        payload.top_k,
        payload.evaluate,
        payload.query_id,
        payload.use_reranker,
        payload.sort_by,
    )

    return RAGResponse(
        status="success",
        mode="text",
        language=target_lang,
        query=query,
        transcription=None,
        answer=result["answer"],
        sources=format_sources(result["documents"], lang=target_lang),
        latency=build_latency(result, eval_ms=result.get("eval_ms")),
        evaluation=format_evaluation(result.get("evaluation")),
        trace_id=result.get("trace_id"),
    )


@router.post("/api/v1/query/voice", response_model=RAGResponse, tags=["Voice"])
async def query_by_voice(
    file: UploadFile = File(..., description="Audio file in WAV, MP3, M4A, OGG, WebM, or FLAC format."),
    language: str = Form(default="gu", description="Language code ('gu', 'hi', or 'auto')"),
    top_k: int = Form(default=5, ge=1, le=20),
    use_reranker: bool = Form(default=False),
    sort_by: str = Form(default="rrf"),
    auto_detect_language: bool = Form(default=True),
    evaluate: bool = Form(default=False),
    query_id: Optional[int] = Form(default=None),
    router_instance: LanguageRouter = Depends(get_router),
    voice_service: VoiceService = Depends(get_voice_service)
):
    """
    Primary Voice RAG Endpoint:
    1. Transcribes audio via Sarvam AI (saaras:v3) or Groq Whisper fallback.
    2. Smart auto-routing based on transcribed text script.
    3. Hybrid RAG retrieval + sorting + optional re-ranking.
    4. LLM answer generation in detected language.
    5. Comprehensive latency breakdown: STT + Retrieval + LLM + Eval + Total.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    target_lang = router_instance.resolve_language(requested_lang=language, auto_detect=False)
    pipeline = router_instance.get_pipeline(target_lang)

    result = await run_in_threadpool(
        handle_voice_query,
        pipeline,
        voice_service,
        audio_bytes,
        file.filename or "input.wav",
        target_lang,
        top_k,
        evaluate,
        query_id,
        use_reranker,
        sort_by,
    )

    if not result.get("transcription"):
        raise HTTPException(
            status_code=422,
            detail=f"Could not transcribe audio into text. Please speak clearly in {pipeline.meta['name']}."
        )

    # If auto detection is active, check if transcription script matches pipeline
    transcription = result["transcription"]
    if auto_detect_language:
        detected_lang = router_instance.resolve_language(query=transcription, requested_lang=language, auto_detect=True)
        if detected_lang != target_lang:
            target_lang = detected_lang
            pipeline = router_instance.get_pipeline(target_lang)
            # Re-run ask with correct pipeline if language was mismatched
            rerun_result = await run_in_threadpool(
                pipeline.ask, transcription, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by
            )
            result.update(rerun_result)

    return RAGResponse(
        status="success",
        mode="voice",
        language=target_lang,
        query=result["transcription"],
        transcription=result["transcription"],
        answer=result["answer"],
        sources=format_sources(result["documents"], lang=target_lang),
        latency=build_latency(result, stt_ms=result.get("stt_ms"), eval_ms=result.get("eval_ms")),
        evaluation=format_evaluation(result.get("evaluation")),
        trace_id=result.get("trace_id"),
    )
