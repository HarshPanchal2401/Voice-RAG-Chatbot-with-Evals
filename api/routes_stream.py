"""
Streaming Routes (SSE & WebSocket)
==================================
- `POST /api/v1/query/text/stream` : Server-Sent Events.
- `WS   /api/v1/voice/live`        : live microphone -> Sarvam realtime STT -> streamed answer + audio.

Both use `api.streaming.stream_answer_events`, so the event order is identical:
token* (interleaved with ordered sentence-level tts*) -> meta/answer -> remaining tts* -> tts_done
-> evaluation -> done.
"""

import asyncio
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

from core.voice_quota import voice_quota, limit_message
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from core.security import PROTECTED_LIMITED, authorize_websocket, ws_max_seconds
from api.dependencies import get_router, get_optional_voice_service, router_from_state, voice_service_from_state
from api.helpers import history_dicts, resolve_index_language, validate_history
from api.streaming import handle_stream_query, stream_answer_events  # noqa: F401  (re-exported for compat)
from schemas.requests import TextQueryRequest, normalize_request_language
from services.voice_service import SARVAM_REALTIME_AUTO_CODE, short_lang_code

router = APIRouter()

SORT_CHOICES = ("rrf", "dense", "sparse", "rerank")
TRAILING_FINAL_WAIT_S = 3.0


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_text(e: BaseException) -> str:
    msg = str(e) or e.__class__.__name__
    return msg[:300]


# ============================================================
# SSE
# ============================================================

@router.post("/api/v1/query/text/stream", tags=["Streaming"], dependencies=PROTECTED_LIMITED)
async def stream_query_by_text(
    payload: TextQueryRequest,
    request: Request,
    router_instance=Depends(get_router),
    voice_service=Depends(get_optional_voice_service),
):
    """
    Server-Sent Events (each `event: <name>\\ndata: <json>\\n\\n`):
    - `token`      {token, language} LLM tokens in real time.
    - `tts`        {type, seq, text, audio_base64, tts_ms} sentence audio in `seq` order, possibly before `meta`.
    - `meta`       {type: "answer", language, answer_language, query, retrieval_query, no_answer, answer, sources, latency, evaluation: null, trace_id}
    - `tts_done`   {type, count, tts_ms} (voice_reply only)
    - `evaluation` {type, evaluation, eval_ms} after the audio (evaluate only)
    - `error`      {error}
    - `done`       {} always last.
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

    speak = bool(payload.voice_reply)
    voice_note = None
    if speak:
        ok, quota = voice_quota.try_consume(request)
        if not ok:
            speak, voice_note = False, limit_message(quota)

    async def sse_event_generator():
        if voice_note:
            yield _sse("voice_limit", {"message": voice_note})
        try:
            async for name, data in stream_answer_events(
                pipeline=pipeline,
                query=query,
                index_lang=target_lang,
                top_k=payload.top_k,
                use_reranker=payload.use_reranker,
                sort_by=payload.sort_by,
                history=history,
                evaluate=payload.evaluate,
                query_id=payload.query_id,
                voice_reply=speak,
                voice_service=voice_service,
                mode="text",
                is_disconnected=request.is_disconnected,
            ):
                if name == "token":
                    yield _sse("token", {"token": data["content"], "language": target_lang})
                else:
                    yield _sse(name, data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️ SSE stream error: {e}")
            yield _sse("error", {"error": _error_text(e)})
        yield _sse("done", {})

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# Live voice WebSocket
# ============================================================

def _create_sarvam_client(api_key: str):
    """Factory (monkeypatched in tests)."""
    from sarvamai import AsyncSarvamAI

    return AsyncSarvamAI(api_subscription_key=api_key)


def _realtime_messages():
    from sarvamai import RealtimeAudioInput, RealtimeEnd

    return RealtimeAudioInput, RealtimeEnd


async def _safe_send(websocket: WebSocket, data: Dict[str, Any]) -> bool:
    try:
        await websocket.send_json(data)
        return True
    except Exception:
        return False


async def _safe_close(websocket: WebSocket, code: int = 1000, reason: str = "") -> None:
    try:
        await websocket.close(code=code, reason=reason[:120] if reason else None)
    except Exception:
        pass


async def _fail(websocket: WebSocket, message: str, code: int = 1011) -> None:
    await _safe_send(websocket, {"type": "error", "message": message})
    await _safe_send(websocket, {"type": "done"})
    await _safe_close(websocket, code=code, reason=message)


class _LiveSTTResult:
    def __init__(self):
        self.finals: List[str] = []
        self.final_langs: List[Optional[str]] = []
        self.latest_partial = ""
        self.stt_ms = 0.0
        self.history: Optional[List[Dict[str, str]]] = None
        self.client_gone = False
        self.timed_out = False


async def _collect_transcript(
    websocket: WebSocket,
    sarvam_api_key: str,
    language_code: str,
    display_lang: str,
    max_seconds: float,
) -> _LiveSTTResult:
    """
    Forwards client PCM frames to Sarvam realtime STT and relays partial/final transcripts.
    Collects *all* final transcripts until the client sends {"type":"stop"} (or disconnects, or the
    session limit is reached), then waits up to TRAILING_FINAL_WAIT_S for Sarvam's trailing finals.
    """
    RealtimeAudioInput, RealtimeEnd = _realtime_messages()
    res = _LiveSTTResult()
    marks = {"stop": None, "last_audio": time.perf_counter()}
    client = _create_sarvam_client(sarvam_api_key)

    async with client.speech_to_text_realtime_streaming.connect(
        language_code=language_code,
        stream_type="fast",
    ) as sarvam_ws:

        async def forward_audio() -> None:
            while True:
                try:
                    msg = await websocket.receive()
                except (WebSocketDisconnect, RuntimeError):
                    res.client_gone = True
                    return
                if msg.get("type") == "websocket.disconnect":
                    res.client_gone = True
                    return
                if msg.get("bytes"):
                    marks["last_audio"] = time.perf_counter()
                    b64 = base64.b64encode(msg["bytes"]).decode("ascii")
                    await sarvam_ws.send_realtime_audio_input(RealtimeAudioInput(audio=b64))
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    if data.get("type") == "history":
                        try:
                            res.history = validate_history(data.get("history"))
                        except ValueError as e:
                            await _safe_send(websocket, {"type": "error", "message": f"Ignoring history: {e}"})
                    elif data.get("type") == "stop":
                        return

        def lang_of(msg) -> str:
            return short_lang_code(getattr(msg, "language", None)) or display_lang

        async def receive_from_sarvam() -> None:
            try:
                async for msg in sarvam_ws:
                    event = getattr(msg, "event", None)
                    if event == "transcript.partial":
                        text = (getattr(msg, "text", "") or "").strip()
                        if text:
                            res.latest_partial = text
                            await _safe_send(websocket, {"type": "partial", "text": text, "language": lang_of(msg)})
                    elif event == "transcript.final":
                        text = (getattr(msg, "text", "") or "").strip()
                        if text:
                            res.finals.append(text)
                            res.final_langs.append(short_lang_code(getattr(msg, "language", None)))
                            res.latest_partial = ""
                            t_end = marks["stop"] or marks["last_audio"]
                            res.stt_ms = max(0.0, (time.perf_counter() - t_end) * 1000)
                            await _safe_send(websocket, {
                                "type": "final",
                                "text": text,
                                "stt_ms": round(res.stt_ms, 2),
                                "language": lang_of(msg),
                            })
                    elif event == "error":
                        print(f"⚠️ Sarvam STT event: {getattr(msg, 'message', msg)}")
                        if getattr(msg, "is_fatal", False):
                            return
                    elif event == "session.end":
                        return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"⚠️ Sarvam receive warning: {e}")

        send_task = asyncio.create_task(forward_audio())
        recv_task = asyncio.create_task(receive_from_sarvam())
        try:
            done, _ = await asyncio.wait({send_task, recv_task}, timeout=max_seconds, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                res.timed_out = True
                print(f"⏱️ Live voice session reached RAG_WS_MAX_SECONDS={max_seconds:g}s; answering what was heard.")
            if not send_task.done():
                send_task.cancel()
            marks["stop"] = time.perf_counter()

            if not res.client_gone and not recv_task.done():
                try:
                    await sarvam_ws.send_realtime_end(RealtimeEnd())
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(asyncio.shield(recv_task), timeout=TRAILING_FINAL_WAIT_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            for task in (send_task, recv_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)

    if not res.finals and res.latest_partial:
        res.finals.append(res.latest_partial)
        res.final_langs.append(None)
        t_end = marks["stop"] or marks["last_audio"]
        res.stt_ms = max(0.0, (time.perf_counter() - t_end) * 1000)
        if not res.client_gone:
            await _safe_send(websocket, {"type": "final", "text": res.latest_partial, "stt_ms": round(res.stt_ms, 2), "language": display_lang})
    return res


def _majority_lang(langs: List[Optional[str]]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for lang in langs:
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=counts.get) if counts else None


@router.websocket("/api/v1/voice/live")
async def live_voice_websocket(
    websocket: WebSocket,
    lang: str = Query(default="gu"),
    evaluate: bool = Query(default=False),
    use_reranker: bool = Query(default=False),
    auto_detect: bool = Query(default=True),
    top_k: int = Query(default=5),
    sort_by: str = Query(default="rrf"),
    voice_reply: bool = Query(default=True),
):
    """
    Live voice: client streams 16 kHz mono PCM16 frames (optional first text frame
    `{"type":"history","history":[...]}`), then `{"type":"stop"}`.
    Server -> client JSON: partial, final, token, tts, answer, tts_done, evaluation, error, done.
    """
    if not await authorize_websocket(websocket):
        return

    try:
        router_instance = router_from_state(websocket.app)
        if router_instance is None:
            await _fail(websocket, "RAG pipelines are not loaded yet. Retry shortly.", code=1013)
            return
        try:
            requested = normalize_request_language(lang)
        except ValueError as e:
            await _fail(websocket, str(e), code=1008)
            return
        sarvam_api_key = os.environ.get("SARVAM_API_KEY", "").strip().strip('"')
        if not sarvam_api_key:
            await _fail(websocket, "SARVAM_API_KEY is not configured on the server.")
            return
        ok, quota = voice_quota.try_consume(websocket)
        if not ok:
            await _fail(websocket, limit_message(quota), code=4429)
            return

        top_k = min(max(int(top_k), 1), 20)
        sort_by = sort_by if sort_by in SORT_CHOICES else "rrf"
        auto = auto_detect or requested == "auto"
        requested_index = router_instance.resolve_language(requested_lang=requested, auto_detect=False)
        locale = (
            SARVAM_REALTIME_AUTO_CODE
            if auto
            else router_instance.get_pipeline(requested_index).meta.get("locale", f"{requested_index}-IN")
        )

        # ---- 1. listen ----
        try:
            stt = await _collect_transcript(websocket, sarvam_api_key, locale, requested_index, ws_max_seconds())
        except Exception as e:
            print(f"⚠️ Live Sarvam STT error: {e}")
            await _fail(websocket, f"Speech recognition failed: {_error_text(e)}")
            return
        if stt.client_gone:
            return
        if not stt.finals:
            await _fail(websocket, "No speech detected. Please speak into the microphone.", code=1000)
            return

        # ---- 2. route by what was actually said ----
        full_query = " ".join(stt.finals).strip()
        stt_lang = _majority_lang(stt.final_langs)
        index_lang = resolve_index_language(router_instance, full_query, requested, auto, stt_lang=stt_lang)
        pipeline = router_instance.get_pipeline(index_lang)
        lang_hint = "en" if stt_lang == "en" else None

        # ---- 3. answer: tokens + sentence audio, then evaluation ----
        gone = asyncio.Event()

        async def watch_client() -> None:
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
            except Exception:
                pass
            gone.set()

        async def is_gone() -> bool:
            return gone.is_set()

        watcher = asyncio.create_task(watch_client())
        try:
            async for name, data in stream_answer_events(
                pipeline=pipeline,
                query=full_query,
                index_lang=index_lang,
                top_k=top_k,
                use_reranker=use_reranker,
                sort_by=sort_by,
                history=stt.history,
                evaluate=evaluate,
                query_id=None,
                voice_reply=voice_reply,
                voice_service=voice_service_from_state(websocket.app),
                stt_ms=stt.stt_ms,
                mode="live_voice",
                lang_hint=lang_hint,
                is_disconnected=is_gone,
            ):
                if name == "token":
                    message = {"type": "token", "content": data["content"]}
                elif name == "meta":
                    message = data  # type == "answer"
                else:
                    message = data
                if not await _safe_send(websocket, message):
                    gone.set()
                    break
        except Exception as e:
            print(f"⚠️ Live voice answer error: {e}")
            await _safe_send(websocket, {"type": "error", "message": _error_text(e)})
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

        if not gone.is_set():
            await _safe_send(websocket, {"type": "done"})
    finally:
        await _safe_close(websocket)
