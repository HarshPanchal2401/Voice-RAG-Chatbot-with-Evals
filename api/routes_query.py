"""
Query Routes (Text & Voice)
===========================
Synchronous Text & Voice RAG endpoints and on-demand TTS.

Voice flow: STT once (provider auto-detects the spoken language when `auto_detect_language`)
-> resolve the index language from the transcript -> RAG once -> evaluation and TTS in parallel.
"""

import asyncio
import base64
from typing import Literal, Optional

from core.voice_quota import voice_quota, limit_message
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Request
from starlette.concurrency import run_in_threadpool

from core.security import (
    ALLOWED_AUDIO_EXTENSIONS,
    PROTECTED, PROTECTED_LIMITED,
    max_upload_bytes,
    too_large_detail,
)
from core.tracing import traceable, add_run_metadata
from api.dependencies import get_router, get_voice_service, get_optional_voice_service
from api.helpers import (
    answer_language_for,
    audio_filename,
    build_latency,
    format_evaluation,
    format_sources,
    history_dicts,
    parse_history_json,
    read_upload_limited,
    resolve_index_language,
    run_evaluation,
)
from schemas.requests import TextQueryRequest, TTSRequest, normalize_request_language
from schemas.responses import RAGResponse, TTSResponse
from services.voice_service import DEFAULT_SPEAKER

router = APIRouter()


def _drop_objects(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k not in ("pipeline", "voice_service", "router_instance", "audio_bytes")}


def _voice_inputs(inputs: dict) -> dict:
    audio = inputs.get("audio_bytes") or b""
    return _drop_objects(inputs) | {"audio_kb": round(len(audio) / 1024, 1)}


def _rag_outputs(outputs):
    if isinstance(outputs, dict) and "documents" in outputs:
        return {k: v for k, v in outputs.items() if k != "documents"} | {"num_sources": len(outputs["documents"])}
    return outputs


@traceable(run_type="chain", name="text_query", process_inputs=_drop_objects, process_outputs=_rag_outputs)
def handle_text_query(
    pipeline,
    query: str,
    lang: str,
    top_k: int,
    use_reranker: bool = False,
    sort_by: str = "rrf",
    history=None,
) -> dict:
    """One RAG run (retrieval + generation). Evaluation / TTS happen afterwards in the route."""
    rag_result = pipeline.ask(query, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by, history=history)
    add_run_metadata(
        language=lang,
        answer_language=rag_result.get("answer_language"),
        mode="text",
        total_ms=round(rag_result.get("total_ms") or 0.0, 2),
        reranked=use_reranker,
        sort_by=sort_by,
        history_turns=len(history or []),
    )
    return rag_result


@traceable(run_type="chain", name="voice_query", process_inputs=_voice_inputs, process_outputs=_rag_outputs)
def handle_voice_query(
    router_instance,
    voice_service,
    audio_bytes: bytes,
    filename: str,
    requested_lang: str,
    auto_detect: bool,
    top_k: int,
    use_reranker: bool = False,
    sort_by: str = "rrf",
    history=None,
) -> dict:
    """STT once -> language from the transcript -> RAG once (all inside one trace)."""
    stt = voice_service.transcribe(
        audio_bytes,
        filename=filename,
        language=requested_lang,
        auto_detect=auto_detect or requested_lang == "auto",
    )
    transcription = (stt.get("text") or "").strip()
    if not transcription:
        return {"transcription": "", "stt_ms": stt.get("stt_ms")}

    stt_lang = stt.get("language")
    index_lang = resolve_index_language(router_instance, transcription, requested_lang, auto_detect, stt_lang=stt_lang)
    pipeline = router_instance.get_pipeline(index_lang)
    rag_result = pipeline.ask(transcription, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by, history=history)
    add_run_metadata(
        language=index_lang,
        answer_language=rag_result.get("answer_language"),
        stt_language=stt_lang,
        mode="voice",
        stt_ms=round(stt.get("stt_ms") or 0.0, 2),
        total_ms=round((stt.get("stt_ms") or 0.0) + (rag_result.get("total_ms") or 0.0), 2),
        reranked=use_reranker,
        sort_by=sort_by,
    )
    return {
        **rag_result,
        "transcription": transcription,
        "stt_ms": stt.get("stt_ms"),
        "stt_language": stt_lang,
        "index_language": index_lang,
    }


async def _evaluate_and_speak(
    pipeline,
    query: str,
    result: dict,
    query_id: Optional[int],
    evaluate: bool,
    voice_service,
    voice_reply: bool,
    tts_lang: str,
):
    """Runs DeepEval and TTS concurrently (both only need the finished answer)."""

    async def _eval():
        if not (evaluate and getattr(pipeline, "evaluator", None)):
            return None, None
        return await run_in_threadpool(run_evaluation, pipeline, query, result, query_id, result.get("trace_id"))

    async def _tts():
        if not (voice_reply and voice_service is not None and result.get("answer")):
            return None, None
        try:
            audio, tts_ms = await run_in_threadpool(voice_service.generate_tts_audio, result["answer"], tts_lang, DEFAULT_SPEAKER, 1.0)
        except Exception as e:
            print(f"⚠️ TTS failed: {e}")
            return None, None
        return (base64.b64encode(audio).decode("ascii") if audio else None), tts_ms

    (eval_result, eval_ms), (audio_b64, tts_ms) = await asyncio.gather(_eval(), _tts())
    return eval_result, eval_ms, audio_b64, tts_ms


@router.post("/api/v1/query/text", response_model=RAGResponse, tags=["Query"], dependencies=PROTECTED_LIMITED)
@router.post("/api/v1/ask", response_model=RAGResponse, tags=["Query"], dependencies=PROTECTED_LIMITED)
async def query_by_text(
    payload: TextQueryRequest,
    request: Request,
    router_instance=Depends(get_router),
    voice_service=Depends(get_optional_voice_service),
):
    """
    Text RAG: language routing (script auto-detection) -> hybrid retrieval (+ optional re-ranking)
    -> grounded answer (history-aware) -> optional DeepEval and Sarvam Bulbul v3 voice reply (in parallel).
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
    history = history_dicts(payload.history)

    result = await run_in_threadpool(
        handle_text_query, pipeline, query, target_lang, payload.top_k, payload.use_reranker, payload.sort_by, history
    )
    answer_lang = answer_language_for(result, target_lang)
    speak = bool(payload.voice_reply and voice_quota.try_consume(request)[0])
    eval_result, eval_ms, audio_b64, tts_ms = await _evaluate_and_speak(
        pipeline, query, result, payload.query_id, payload.evaluate, voice_service, speak, answer_lang
    )

    return RAGResponse(
        status="success",
        mode="text",
        language=result.get("language") or target_lang,
        answer_language=answer_lang,
        query=query,
        retrieval_query=result.get("retrieval_query") or query,
        transcription=None,
        answer=result.get("answer", ""),
        no_answer=bool(result.get("no_answer", False)),
        sources=format_sources(result.get("documents") or [], lang=target_lang),
        latency=build_latency(result, eval_ms=eval_ms, tts_ms=tts_ms),
        evaluation=format_evaluation(eval_result),
        audio_base64=audio_b64,
        trace_id=result.get("trace_id"),
    )


@router.post("/api/v1/query/voice", response_model=RAGResponse, tags=["Voice"], dependencies=PROTECTED_LIMITED)
async def query_by_voice(
    file: UploadFile = File(..., description="Audio file: wav, mp3, m4a, ogg, webm, flac or mp4 (max RAG_MAX_UPLOAD_MB)."),
    language: str = Form(default="gu", description="'gu', 'hi' or 'auto'"),
    top_k: int = Form(default=5, ge=1, le=20),
    use_reranker: bool = Form(default=False),
    sort_by: Literal["rrf", "dense", "sparse", "rerank"] = Form(default="rrf"),
    auto_detect_language: bool = Form(default=True, description="Let STT detect the spoken language and route by it."),
    evaluate: bool = Form(default=False),
    voice_reply: bool = Form(default=True, description="Reply with Sarvam Bulbul v3 speech."),
    query_id: Optional[int] = Form(default=None),
    history_json: Optional[str] = Form(default=None, description="JSON array of {role, content} previous turns."),
    request: Request = None,
    router_instance=Depends(get_router),
    voice_service=Depends(get_voice_service),
):
    """
    Voice RAG: STT once (Sarvam saaras:v3, Groq Whisper fallback) -> language from the transcript
    -> RAG once -> DeepEval + Bulbul v3 TTS in parallel. Returns one concatenated MP3 in `audio_base64`.
    """
    try:
        requested = normalize_request_language(language)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    allowed, quota = voice_quota.try_consume(request)
    if not allowed:
        raise HTTPException(status_code=429, detail=limit_message(quota), headers={"Retry-After": str(quota["reset_in_seconds"] or 3600)})
    filename = audio_filename(file, ALLOWED_AUDIO_EXTENSIONS)
    history = parse_history_json(history_json)
    audio_bytes = await read_upload_limited(file, max_upload_bytes(), too_large_detail())
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    result = await run_in_threadpool(
        handle_voice_query,
        router_instance,
        voice_service,
        audio_bytes,
        filename,
        requested,
        auto_detect_language,
        top_k,
        use_reranker,
        sort_by,
        history,
    )
    transcription = result.get("transcription")
    if not transcription:
        raise HTTPException(status_code=422, detail="Could not transcribe the audio. Please speak clearly and try again.")

    index_lang = result["index_language"]
    pipeline = router_instance.get_pipeline(index_lang)
    answer_lang = answer_language_for(result, index_lang, stt_lang=result.get("stt_language"))
    eval_result, eval_ms, audio_b64, tts_ms = await _evaluate_and_speak(
        pipeline, transcription, result, query_id, evaluate, voice_service, voice_reply, answer_lang
    )

    return RAGResponse(
        status="success",
        mode="voice",
        language=result.get("language") or index_lang,
        answer_language=answer_lang,
        query=transcription,
        retrieval_query=result.get("retrieval_query") or transcription,
        transcription=transcription,
        answer=result.get("answer", ""),
        no_answer=bool(result.get("no_answer", False)),
        sources=format_sources(result.get("documents") or [], lang=index_lang),
        latency=build_latency(result, stt_ms=result.get("stt_ms"), eval_ms=eval_ms, tts_ms=tts_ms),
        evaluation=format_evaluation(eval_result),
        audio_base64=audio_b64,
        trace_id=result.get("trace_id"),
    )


@router.post("/api/v1/voice/tts", response_model=TTSResponse, tags=["Voice"], dependencies=PROTECTED_LIMITED)
async def convert_text_to_speech(
    payload: TTSRequest,
    request: Request,
    voice_service=Depends(get_voice_service),
):
    """On-demand TTS (Gujarati, Hindi or English) with Sarvam bulbul:v3, sentence-chunked and capped."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    allowed, quota = voice_quota.try_consume(request)
    if not allowed:
        raise HTTPException(status_code=429, detail=limit_message(quota), headers={"Retry-After": str(quota["reset_in_seconds"] or 3600)})

    speaker = payload.speaker or DEFAULT_SPEAKER
    res = await run_in_threadpool(voice_service.generate_tts, text, payload.language, speaker, payload.pace or 1.0)
    if not res.get("audio"):
        raise HTTPException(status_code=502, detail="Could not synthesize speech audio. Please verify SARVAM_API_KEY.")

    return TTSResponse(
        status="success",
        language=payload.language,
        speaker=speaker,
        audio_base64=base64.b64encode(res["audio"]).decode("ascii"),
        tts_ms=round(res.get("tts_ms") or 0.0, 2),
        chunks=res.get("chunks"),
        spoken_chars=res.get("spoken_chars"),
    )


@router.get("/api/v1/voice/quota", tags=["Voice"], dependencies=PROTECTED)
async def get_voice_quota(request: Request):
    """Remaining Sarvam voice uses for this user (browser id + IP)."""
    return voice_quota.status(request)
