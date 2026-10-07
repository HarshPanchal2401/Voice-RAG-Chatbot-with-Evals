"""
API Route Helpers
=================
Conversion of pipeline outputs into typed Pydantic models, language resolution helpers,
history parsing, upload reading and evaluation glue shared by the routers.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile
from pydantic import TypeAdapter, ValidationError

from core.tracing import log_eval_feedback
from schemas.models import (
    ChatTurn,
    SourceDocument,
    EvaluationScores,
    EvaluationResult,
    RetrievalTimings,
    LatencyBreakdown,
)
from schemas.requests import MAX_HISTORY_TURNS
from services.voice_service import script_language, short_lang_code


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, digits: Optional[int] = None) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if digits is not None else f


# ------------------------------------------------------------
# Pipeline output -> response models
# ------------------------------------------------------------

def format_sources(raw_docs: list, lang: str = "gu") -> List[SourceDocument]:
    """Converts raw document dicts into validated SourceDocument instances."""
    formatted = []
    for rank, doc in enumerate(raw_docs or [], start=1):
        source_query_id = _as_int(doc.get("query_id"))
        passage_id = _as_int(doc.get("passage_id")) or 0
        doc_key = doc.get("doc_key") or (f"{source_query_id}:{passage_id}" if source_query_id is not None else None)
        formatted.append(SourceDocument(
            rank=_as_int(doc.get("rank")) or rank,
            vector_id=_as_int(doc.get("vector_id")),
            source_query_id=source_query_id,
            doc_key=doc_key,
            chunk_id=_as_int(doc.get("chunk_id")) or 0,
            passage_id=passage_id,
            part_id=_as_int(doc.get("part_id")),
            url=doc.get("url"),
            title=doc.get("title"),
            section=doc.get("section"),
            text=doc.get("text") or "",
            language=doc.get("language") or lang,
            rrf_score=_as_float(doc.get("rrf_score"), 6) or 0.0,
            dense_rank=_as_int(doc.get("dense_rank")),
            sparse_rank=_as_int(doc.get("sparse_rank")),
            dense_score=_as_float(doc.get("dense_score"), 4),
            sparse_score=_as_float(doc.get("sparse_score"), 4),
            rerank_score=_as_float(doc.get("rerank_score"), 4),
            rerank_rank=_as_int(doc.get("rerank_rank")),
            original_rank=_as_int(doc.get("original_rank")),
            rerank_reason=doc.get("rerank_reason"),
        ))
    return formatted


def format_evaluation(eval_raw: Optional[dict]) -> Optional[EvaluationResult]:
    """Converts the evaluator's dict into EvaluationResult."""
    if not eval_raw:
        return None

    scores = None
    if eval_raw.get("scores"):
        s = eval_raw["scores"]
        scores = EvaluationScores(
            overall_score=_as_float(s.get("overall_score")),
            faithfulness=_as_float(s.get("faithfulness")),
            answer_relevance=_as_float(s.get("answer_relevance")),
            answer_correctness=_as_float(s.get("answer_correctness")),
            answer_similarity=_as_float(s.get("answer_similarity")),
            context_recall=_as_float(s.get("context_recall")),
            hit_rate_at_k=_as_float(s.get("hit_rate_at_k")),
            explanation=s.get("explanation"),
            reason=s.get("reason"),
        )

    failed = eval_raw.get("failed_metrics")
    return EvaluationResult(
        is_golden=bool(eval_raw.get("is_golden", False)),
        query_id=_as_int(eval_raw.get("query_id")),
        query_type=eval_raw.get("query_type"),
        ground_truth_answer=eval_raw.get("ground_truth_answer"),
        warning=eval_raw.get("warning"),
        scores=scores,
        failed_metrics=[str(m) for m in failed] if isinstance(failed, (list, tuple)) else None,
    )


def build_retrieval_timings(raw: Optional[dict]) -> RetrievalTimings:
    """Known timing keys only; None / non-numeric values fall back to the field default."""
    raw = raw or {}
    clean = {}
    for key in RetrievalTimings.model_fields:
        value = _as_float(raw.get(key))
        if value is not None:
            clean[key] = value
    return RetrievalTimings(**clean)


def build_latency(
    rag_result: dict,
    stt_ms: Optional[float] = None,
    eval_ms: Optional[float] = None,
    tts_ms: Optional[float] = None,
) -> LatencyBreakdown:
    """Builds the LatencyBreakdown model (total = STT + retrieval/LLM + sequential TTS)."""
    timings = rag_result.get("retrieval_timings") or rag_result.get("timings") or {}
    rag_total = _as_float(rag_result.get("total_ms")) or 0.0
    return LatencyBreakdown(
        stt_ms=_as_float(stt_ms, 2),
        retrieval=build_retrieval_timings(timings),
        ttft_ms=_as_float(rag_result.get("ttft_ms")),
        llm_ms=_as_float(rag_result.get("llm_ms")) or 0.0,
        eval_ms=_as_float(eval_ms, 2),
        tts_ms=_as_float(tts_ms, 2),
        total_ms=round((stt_ms or 0.0) + rag_total + (tts_ms or 0.0), 2),
    )


def sanitize_sample_queries(samples: list) -> List[Dict[str, Any]]:
    """Only question / query_type / query_id are exposed (never ground-truth answers or contexts)."""
    out = []
    for s in samples or []:
        if not isinstance(s, dict):
            continue
        question = s.get("question") or s.get("query")
        if not question:
            continue
        out.append({"question": question, "query_type": s.get("query_type"), "query_id": _as_int(s.get("query_id"))})
    return out


# ------------------------------------------------------------
# Language resolution
# ------------------------------------------------------------

def resolve_index_language(
    router,
    text: str,
    requested: str,
    auto_detect: bool,
    stt_lang: Optional[str] = None,
) -> str:
    """
    Index (corpus) language for a transcript / query:
    1. auto-detect on -> script of the text (Gujarati vs Devanagari) when a pipeline exists for it;
    2. else the language the STT provider detected (gu/hi);
    3. else the requested language ('auto' -> router default).
    An English transcript keeps the requested index; the answer language is handled by the pipeline.
    """
    stt_short = short_lang_code(stt_lang) if stt_lang else None
    fallback = requested
    if auto_detect and stt_short in ("gu", "hi") and stt_short in getattr(router, "pipelines", {}):
        fallback = stt_short
    if (fallback or "").lower() == "auto":
        fallback = None
    return router.resolve_language(query=text, requested_lang=fallback or "auto", auto_detect=auto_detect or requested == "auto")


def answer_language_for(meta: dict, index_lang: str, stt_lang: Optional[str] = None) -> str:
    """Answer language: the pipeline's `answer_language`, else 'en' for English speech, else the script / index."""
    lang = meta.get("answer_language")
    if lang in ("gu", "hi", "en"):
        return lang
    if short_lang_code(stt_lang) == "en":
        return "en"
    return script_language(meta.get("answer") or "") or index_lang


# ------------------------------------------------------------
# History
# ------------------------------------------------------------

_history_adapter = TypeAdapter(List[ChatTurn])


def history_dicts(history: Optional[List[ChatTurn]]) -> Optional[List[Dict[str, str]]]:
    if not history:
        return None
    return [{"role": t.role, "content": t.content} for t in history]


def validate_history(raw: Any) -> Optional[List[Dict[str, str]]]:
    """Validates a raw history list (from JSON). Raises ValueError with a readable message."""
    if raw in (None, "", []):
        return None
    if not isinstance(raw, list):
        raise ValueError("history must be a list of {role, content} objects.")
    if len(raw) > MAX_HISTORY_TURNS:
        raise ValueError(f"history may contain at most {MAX_HISTORY_TURNS} messages.")
    try:
        turns = _history_adapter.validate_python(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid history: {e.errors(include_url=False)[0].get('msg', 'invalid item')}") from e
    return history_dicts(turns)


def parse_history_json(history_json: Optional[str]) -> Optional[List[Dict[str, str]]]:
    """Parses the multipart `history_json` field; 422 on malformed input."""
    if history_json is None or not history_json.strip():
        return None
    try:
        raw = json.loads(history_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="history_json must be a JSON array of {role, content} objects.")
    try:
        return validate_history(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ------------------------------------------------------------
# Uploads
# ------------------------------------------------------------

_CONTENT_TYPE_EXT = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/vnd.wave": ".wav",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/m4a": ".m4a", "video/mp4": ".mp4",
    "audio/ogg": ".ogg", "audio/webm": ".webm", "video/webm": ".webm",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
}


def audio_filename(file: UploadFile, allowed_exts: Tuple[str, ...]) -> str:
    """Returns a filename with an allowed audio extension, or raises 415."""
    name = os.path.basename(file.filename or "")
    ext = os.path.splitext(name)[1].lower()
    if ext in allowed_exts:
        return name
    if not ext:
        ctype = (file.content_type or "").split(";")[0].strip().lower()
        guessed = _CONTENT_TYPE_EXT.get(ctype)
        if guessed in allowed_exts:
            return f"{name or 'audio'}{guessed}"
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported audio format '{ext or file.content_type or 'unknown'}'. Allowed: {', '.join(e.lstrip('.') for e in allowed_exts)}.",
    )


async def read_upload_limited(file: UploadFile, max_bytes: int, too_large_detail: str) -> bytes:
    """Reads an upload in 1 MB chunks and raises 413 as soon as it exceeds `max_bytes`."""
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=too_large_detail)
        chunks.append(chunk)
    return b"".join(chunks)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def run_evaluation(
    pipeline,
    query: str,
    rag_result: dict,
    query_id: Optional[int],
    trace_run_id: Optional[str],
) -> tuple:
    """DeepEval scoring + LangSmith feedback on the request trace. Returns (eval_result, eval_ms)."""
    t_eval = time.perf_counter()
    try:
        eval_result = pipeline.evaluate(
            query=query,
            answer=rag_result.get("answer", ""),
            documents=rag_result.get("documents") or [],
            query_id=query_id,
            allow_open_eval=True,
        )
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        return None, None
    eval_ms = (time.perf_counter() - t_eval) * 1000
    log_eval_feedback(trace_run_id, eval_result)
    return eval_result, eval_ms
