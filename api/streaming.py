"""
Streaming Answer Orchestration (shared by SSE and the live-voice WebSocket)
===========================================================================
Runs `pipeline.ask_stream` in a worker thread and turns it into an ordered async event stream:

    token* (interleaved with) tts*  ->  meta  ->  tts* (remaining)  ->  tts_done  ->  evaluation

- Sentence-level TTS: as tokens arrive, complete sentences (>= RAG_TTS_MIN_CHUNK_CHARS, default 40)
  are cut and submitted to the TTS thread pool immediately. `tts` events are emitted strictly in
  `seq` order as soon as the next chunk is ready, so audio can start while the LLM is still writing.
- Evaluation is started only after the answer is complete and is emitted after `tts_done`; it never
  delays audio.
- The caller passes `is_disconnected`; when the client goes away the LLM stream is stopped, queued
  TTS requests are cancelled and no evaluation is started.

Event tuples yielded: (name, payload) with names
`token` {"content"}, `tts`, `meta`, `tts_done`, `evaluation`; errors propagate as exceptions.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

from starlette.concurrency import run_in_threadpool

from core.tracing import traceable, add_run_metadata
from api.helpers import build_latency, format_evaluation, format_sources, run_evaluation
from services.voice_service import (
    DEFAULT_SPEAKER,
    clean_text_for_speech,
    script_language,
    split_long_text,
    tts_chunk_chars,
    tts_max_chars,
)

_BOUNDARY_RE = re.compile(r"[.?!।॥]+(?=\s)|\n")


def tts_min_chunk_chars() -> int:
    try:
        return max(1, int(os.environ.get("RAG_TTS_MIN_CHUNK_CHARS", "") or 40))
    except ValueError:
        return 40


# ------------------------------------------------------------
# Incremental sentence chunker
# ------------------------------------------------------------

class IncrementalSentenceChunker:
    """
    Accumulates streamed tokens and cuts speakable chunks at sentence boundaries
    (`. ? ! । ॥` followed by whitespace, or a newline). A chunk is emitted once it holds at least
    `min_chars` characters of cleaned text; a run-on sentence is cut at a word boundary once it
    exceeds `max_chars`. The total spoken text is capped at `max_total` at a sentence boundary.
    """

    def __init__(self, min_chars: Optional[int] = None, max_chars: Optional[int] = None, max_total: Optional[int] = None):
        self.min_chars = min_chars or tts_min_chunk_chars()
        self.max_chars = max_chars or tts_chunk_chars()
        self.max_total = max_total or tts_max_chars()
        self.buffer = ""
        self.total = 0
        self.exhausted = False
        self.emitted = 0

    def reset(self, text: str = "") -> None:
        self.buffer = text

    def feed(self, token: str) -> List[str]:
        if self.exhausted:
            return []
        self.buffer += token or ""
        return self._cut(final=False)

    def flush(self) -> List[str]:
        if self.exhausted:
            return []
        return self._cut(final=True)

    def _cut(self, final: bool) -> List[str]:
        out: List[str] = []
        while not self.exhausted and self.buffer:
            cut_at = None
            for m in _BOUNDARY_RE.finditer(self.buffer):
                if len(clean_text_for_speech(self.buffer[:m.end()])) >= self.min_chars:
                    cut_at = m.end()
                    break
            if cut_at is None and len(self.buffer) > self.max_chars:
                space = self.buffer.rfind(" ", 0, self.max_chars)
                cut_at = space if space > 0 else self.max_chars
            if cut_at is None:
                if not final:
                    break
                cut_at = len(self.buffer)
            piece, self.buffer = self.buffer[:cut_at], self.buffer[cut_at:]
            cleaned = re.sub(r"\s+", " ", clean_text_for_speech(piece)).strip()
            if not cleaned or not any(ch.isalnum() for ch in cleaned):
                continue
            accepted = self._accept(cleaned)
            out.extend(accepted)
        return out

    def _accept(self, cleaned: str) -> List[str]:
        pieces = split_long_text(cleaned, self.max_chars) if len(cleaned) > self.max_chars else [cleaned]
        accepted: List[str] = []
        for piece in pieces:
            if self.total + len(piece) > self.max_total:
                if self.total == 0:
                    accepted.append(split_long_text(piece, self.max_total)[0])
                    self.total = self.max_total
                self.exhausted = True
                self.buffer = ""
                break
            accepted.append(piece)
            self.total += len(piece)
        self.emitted += len(accepted)
        return accepted


# ------------------------------------------------------------
# Language of the spoken answer
# ------------------------------------------------------------

_detect_impl: Any = None


def detect_answer_lang(text: str, index_lang: str, hint: Optional[str] = None) -> str:
    """'gu' | 'hi' | 'en' for TTS (uses `pipeline.generator.detect_answer_language` when available)."""
    global _detect_impl
    if _detect_impl is None:
        try:
            from pipeline.generator import detect_answer_language  # type: ignore
            _detect_impl = detect_answer_language
        except Exception:
            _detect_impl = False
    if _detect_impl and text and text.strip():
        try:
            lang = _detect_impl(text, index_lang)
            if lang in ("gu", "hi", "en"):
                return lang
        except Exception:
            pass
    return script_language(text or "") or hint or index_lang


# ------------------------------------------------------------
# Traced producer (runs in a worker thread)
# ------------------------------------------------------------

def _drop_objects(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k not in ("pipeline", "voice_service", "router_instance")}


def _stream_reduce(items: list) -> dict:
    for item in reversed(items or []):
        if isinstance(item, dict) and item.get("type") == "meta":
            return {k: v for k, v in item.items() if k != "documents"} | {"num_sources": len(item.get("documents") or [])}
    return {}


@traceable(run_type="chain", name="stream_query", process_inputs=_drop_objects, reduce_fn=_stream_reduce)
def handle_stream_query(
    pipeline,
    query: str,
    lang: str,
    top_k: int,
    use_reranker: bool = False,
    sort_by: str = "rrf",
    history: Optional[List[Dict[str, str]]] = None,
    stt_ms: Optional[float] = None,
    mode: str = "text",
):
    """Yields the pipeline's {"type": "token"} items and its final {"type": "meta"} item."""
    meta = None
    for chunk in pipeline.ask_stream(query, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by, history=history):
        if chunk.get("type") == "meta":
            meta = chunk
        yield chunk
    if meta is not None:
        add_run_metadata(
            language=lang,
            answer_language=meta.get("answer_language"),
            mode=mode,
            stt_ms=stt_ms,
            ttft_ms=round(meta.get("ttft_ms") or 0.0, 2),
            total_ms=round((stt_ms or 0.0) + (meta.get("total_ms") or 0.0), 2),
            reranked=use_reranker,
            sort_by=sort_by,
            history_turns=len(history or []),
        )


# ------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------

@dataclass
class _TTSJob:
    text: str
    language: str
    future: "asyncio.Future"


@dataclass
class _TTSState:
    jobs: List[_TTSJob] = field(default_factory=list)
    next_idx: int = 0
    emitted: int = 0
    first_submit: Optional[float] = None
    last_done: Optional[float] = None
    first_audio_at: Optional[float] = None


def build_answer_payload(meta: dict, query: str, index_lang: str, stt_ms: Optional[float], lang_hint: Optional[str]) -> dict:
    """The `meta` (SSE) / `answer` (WebSocket) payload."""
    lang = meta.get("language") or index_lang
    latency = build_latency(meta, stt_ms=stt_ms)
    return {
        "type": "answer",
        "language": lang,
        "answer_language": meta.get("answer_language") or lang_hint or lang,
        "query": query,
        "retrieval_query": meta.get("retrieval_query") or query,
        "no_answer": bool(meta.get("no_answer", False)),
        "answer": meta.get("answer", ""),
        "sources": [s.model_dump() for s in format_sources(meta.get("documents") or [], lang=lang)],
        "latency": latency.model_dump(),
        "evaluation": None,
        "trace_id": meta.get("trace_id"),
    }


async def stream_answer_events(
    *,
    pipeline,
    query: str,
    index_lang: str,
    top_k: int = 5,
    use_reranker: bool = False,
    sort_by: str = "rrf",
    history: Optional[List[Dict[str, str]]] = None,
    evaluate: bool = False,
    query_id: Optional[int] = None,
    voice_reply: bool = False,
    voice_service=None,
    stt_ms: Optional[float] = None,
    mode: str = "text",
    lang_hint: Optional[str] = None,
    speaker: str = DEFAULT_SPEAKER,
    pace: float = 1.0,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncIterator[Tuple[str, dict]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    t_start = time.perf_counter()
    tts_on = bool(voice_reply and voice_service is not None and hasattr(voice_service, "synthesize_chunk"))
    chunker = IncrementalSentenceChunker() if tts_on else None
    tts = _TTSState()
    streamed: List[str] = []
    eval_task: Optional[asyncio.Future] = None

    def post(kind: str, payload: Any = None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
        except RuntimeError:
            pass  # event loop closed (client gone / shutdown)

    def produce() -> None:
        gen = handle_stream_query(pipeline, query, index_lang, top_k, use_reranker, sort_by, history, stt_ms, mode)
        try:
            for item in gen:
                if stop.is_set():
                    break
                post("item", item)
        except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
            post("error", e)
            return
        finally:
            try:
                gen.close()
            except Exception:
                pass
        post("end")

    ctx = contextvars.copy_context()
    loop.run_in_executor(None, ctx.run, produce)

    async def gone() -> bool:
        if is_disconnected is None:
            return False
        try:
            return await is_disconnected()
        except Exception:
            return True

    def submit(chunks: List[str], current_text: str, final_lang: Optional[str] = None) -> None:
        executor = getattr(voice_service, "executor", None)
        for chunk in chunks:
            lang = final_lang or detect_answer_lang(current_text, index_lang, lang_hint)
            fut = loop.run_in_executor(executor, voice_service.synthesize_chunk, chunk, lang, speaker, pace)
            fut.add_done_callback(lambda _f: queue.put_nowait(("tts_ready", None)))
            if tts.first_submit is None:
                tts.first_submit = time.perf_counter()
            tts.jobs.append(_TTSJob(text=chunk, language=lang, future=fut))

    def ready_tts_events() -> List[Tuple[str, dict]]:
        events: List[Tuple[str, dict]] = []
        while tts.next_idx < len(tts.jobs) and tts.jobs[tts.next_idx].future.done():
            job = tts.jobs[tts.next_idx]
            tts.next_idx += 1
            tts.last_done = time.perf_counter()
            audio, ms = None, 0.0
            if not job.future.cancelled() and job.future.exception() is None:
                audio, ms = job.future.result()
            if not audio:
                continue
            if tts.first_audio_at is None:
                tts.first_audio_at = time.perf_counter()
            events.append(("tts", {
                "type": "tts",
                "seq": tts.emitted,
                "text": job.text,
                "language": job.language,
                "audio_base64": base64.b64encode(audio).decode("ascii"),
                "tts_ms": round(ms or 0.0, 2),
            }))
            tts.emitted += 1
        return events

    try:
        # ---- 1. tokens (+ sentence TTS as soon as sentences complete) ----
        meta: Optional[dict] = None
        n_events = 0
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if await gone():
                    return
                continue
            n_events += 1
            if n_events % 25 == 0 and await gone():
                return
            if kind == "item":
                if payload.get("type") == "token":
                    content = payload.get("content") or ""
                    if not content:
                        continue
                    streamed.append(content)
                    yield "token", {"content": content}
                    if tts_on:
                        submit(chunker.feed(content), "".join(streamed))
                elif payload.get("type") == "meta":
                    meta = payload
            elif kind == "tts_ready":
                for ev in ready_tts_events():
                    yield ev
            elif kind == "error":
                raise payload
            elif kind == "end":
                break

        if meta is None:
            raise RuntimeError("The pipeline finished without producing an answer.")

        # ---- 2. remaining text -> TTS, then meta ----
        answer = meta.get("answer", "") or ""
        answer_lang = meta.get("answer_language") or detect_answer_lang(answer, index_lang, lang_hint)
        if tts_on:
            if not tts.jobs and "".join(streamed).strip() != answer.strip():
                chunker.reset(answer)  # answer was not streamed as tokens (e.g. canned no-answer text)
            submit(chunker.flush(), answer, final_lang=answer_lang)

        answer_payload = build_answer_payload(meta, query, index_lang, stt_ms, lang_hint)
        answer_payload["answer_language"] = meta.get("answer_language") or answer_lang
        yield "meta", answer_payload

        # Evaluation starts only now (after the answer); it runs in parallel with the remaining TTS
        # but is emitted after `tts_done`, so it never delays audio.
        if evaluate and getattr(pipeline, "evaluator", None) and not await gone():
            eval_task = asyncio.ensure_future(
                run_in_threadpool(run_evaluation, pipeline, query, meta, query_id, meta.get("trace_id"))
            )

        # ---- 3. remaining audio ----
        if voice_reply:
            for ev in ready_tts_events():
                yield ev
            while tts.next_idx < len(tts.jobs):
                try:
                    kind, _ = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await gone():
                        return
                    continue
                if kind == "tts_ready":
                    for ev in ready_tts_events():
                        yield ev
            wall_ms = ((tts.last_done or tts.first_submit or 0.0) - (tts.first_submit or 0.0)) * 1000 if tts.first_submit else 0.0
            yield "tts_done", {
                "type": "tts_done",
                "count": tts.emitted,
                "tts_ms": round(wall_ms, 2),
                "first_audio_ms": round((tts.first_audio_at - t_start) * 1000, 2) if tts.first_audio_at else None,
                "available": tts_on,
            }

        # ---- 4. evaluation ----
        if eval_task is not None:
            eval_result, eval_ms = await eval_task
            evaluation = format_evaluation(eval_result)
            yield "evaluation", {
                "type": "evaluation",
                "evaluation": evaluation.model_dump() if evaluation else None,
                "eval_ms": round(eval_ms, 2) if eval_ms is not None else None,
            }
    finally:
        stop.set()
        for job in tts.jobs:
            if not job.future.done():
                job.future.cancel()
        if eval_task is not None and not eval_task.done():
            eval_task.cancel()
