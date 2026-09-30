"""
Streaming Routes (SSE & WebSocket)
==================================
Real-time token streaming via Server-Sent Events (SSE) and live microphone STT WebSocket.
"""

import os
import json
import time
import base64
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from core.tracing import traceable, current_run_id, add_run_metadata
from pipeline.router import LanguageRouter
from pipeline.single_pipeline import SingleLanguagePipeline
from api.dependencies import get_router, get_voice_service
from services.voice_service import VoiceService
from api.helpers import format_sources, format_evaluation, build_latency, run_evaluation
from schemas.requests import TextQueryRequest

router = APIRouter()


def _drop_objects(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k not in ("pipeline", "voice_service")}


def _stream_reduce(items: list) -> dict:
    return items[-1] if items and isinstance(items[-1], dict) else {}


@traceable(run_type="chain", name="stream_query", process_inputs=_drop_objects, reduce_fn=_stream_reduce)
def handle_stream_query(
    pipeline: SingleLanguagePipeline,
    query: str,
    lang: str,
    top_k: int,
    evaluate: bool,
    query_id: Optional[int],
    stt_ms: Optional[float] = None,
    mode: str = "text",
    use_reranker: bool = False,
    sort_by: str = "rrf",
):
    """
    Yields {"type": "token"} items, then {"type": "answer"} (answer + sources + latency),
    then {"type": "evaluation"} once DeepEval is done.
    """
    root_id = current_run_id()
    meta = None
    for chunk in pipeline.ask_stream(query, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by):
        if chunk["type"] == "token":
            yield chunk
        elif chunk["type"] == "meta":
            meta = chunk

    if meta is None:
        return

    latency = build_latency(meta, stt_ms=stt_ms)
    add_run_metadata(
        language=lang,
        mode=mode,
        stt_ms=stt_ms,
        ttft_ms=round(meta.get("ttft_ms", 0.0), 2),
        total_ms=round(latency.total_ms, 2),
        reranked=use_reranker,
        sort_by=sort_by,
    )

    yield {
        "type": "answer",
        "language": lang,
        "query": query,
        "answer": meta["answer"],
        "sources": [s.model_dump() for s in format_sources(meta["documents"], lang=lang)],
        "latency": latency.model_dump(),
        "evaluation": None,
        "trace_id": meta.get("trace_id"),
    }

    if evaluate and pipeline.evaluator:
        eval_result, eval_ms = run_evaluation(pipeline, query, meta, query_id, root_id)
        yield {
            "type": "evaluation",
            "evaluation": format_evaluation(eval_result).model_dump() if eval_result else None,
            "eval_ms": eval_ms,
        }


@router.post("/api/v1/query/text/stream", tags=["Streaming"])
async def stream_query_by_text(
    payload: TextQueryRequest,
    router_instance: LanguageRouter = Depends(get_router),
    voice_service: VoiceService = Depends(get_voice_service)
):
    """
    Streaming Server-Sent Events (SSE) Endpoint:
    - event: token       LLM tokens as they are generated in real-time.
    - event: meta        full answer, sources and latency profile (sent before evaluation starts).
    - event: tts         synthesized speech audio (if voice_reply=true).
    - event: evaluation  DeepEval scorecard (only when evaluate=true).
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

    async def sse_event_generator():
        try:
            full_answer = ""
            generator = handle_stream_query(
                pipeline,
                query,
                target_lang,
                payload.top_k,
                payload.evaluate,
                payload.query_id,
                use_reranker=payload.use_reranker,
                sort_by=payload.sort_by,
            )
            async for item in iterate_in_threadpool(generator):
                if item["type"] == "token":
                    data_str = json.dumps({"token": item["content"], "language": target_lang}, ensure_ascii=False)
                    yield f"event: token\ndata: {data_str}\n\n"
                elif item["type"] == "answer":
                    full_answer = item.get("answer", "")
                    yield f"event: meta\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                    # Synthesize TTS voice reply if requested
                    if payload.voice_reply and voice_service and full_answer:
                        try:
                            audio_bytes, tts_ms = await run_in_threadpool(
                                voice_service.generate_tts_audio,
                                full_answer,
                                target_lang,
                                "shubh",
                                1.0,
                                22050,
                                "bulbul:v3"
                            )
                            if audio_bytes:
                                tts_payload = {
                                    "type": "tts",
                                    "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                                    "tts_ms": round(tts_ms, 2)
                                }
                                yield f"event: tts\ndata: {json.dumps(tts_payload, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            print(f"⚠️ SSE TTS synthesis warning: {e}")
                elif item["type"] == "evaluation":
                    yield f"event: evaluation\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        except Exception as e:
            err_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.websocket("/api/v1/voice/live")
async def live_voice_websocket(
    websocket: WebSocket,
    lang: str = Query(default="gu"),
    evaluate: bool = Query(default=False),
    use_reranker: bool = Query(default=False)
):
    """
    Real-Time Sarvam AI Live Streaming Speech-to-Text WebSocket:
    1. Client streams live 16kHz PCM audio bytes in chunks.
    2. Server forwards audio chunks to Sarvam AI Realtime Streaming WebSocket.
    3. Emits live 'partial' transcripts back to UI.
    4. Upon 'final' transcript (or stop with partial fallback), streams 'token' messages,
       then 'answer', then 'tts', then 'evaluation'.
    """
    await websocket.accept()
    router_instance: LanguageRouter = websocket.app.state.router
    target_lang = router_instance.resolve_language(requested_lang=lang, auto_detect=False)
    pipeline = router_instance.get_pipeline(target_lang)
    locale_code = pipeline.meta.get("locale", "gu-IN")
    sarvam_api_key = os.environ.get("SARVAM_API_KEY", "").strip()

    if not sarvam_api_key:
        await websocket.send_json({"type": "error", "message": "SARVAM_API_KEY is not configured on server."})
        await websocket.close()
        return

    from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd

    client = AsyncSarvamAI(api_subscription_key=sarvam_api_key)
    received_final_text = []
    latest_partial = ""
    speech_end = {"t": None, "last_audio": time.perf_counter()}
    stt_ms = 0.0

    try:
        async with client.speech_to_text_realtime_streaming.connect(
            language_code=locale_code,
            stream_type="fast",
        ) as sarvam_ws:

            async def forward_audio_to_sarvam():
                try:
                    while True:
                        data = await websocket.receive()
                        if data.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in data and data["bytes"]:
                            speech_end["last_audio"] = time.perf_counter()
                            b64 = base64.b64encode(data["bytes"]).decode("utf-8")
                            await sarvam_ws.send_realtime_audio_input(RealtimeAudioInput(audio=b64))
                        elif "text" in data and data["text"]:
                            try:
                                msg = json.loads(data["text"])
                            except Exception:
                                msg = {}
                            if msg.get("type") == "stop":
                                speech_end["t"] = time.perf_counter()
                                try:
                                    await sarvam_ws.send_realtime_end(RealtimeEnd())
                                except Exception:
                                    pass
                                break
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    print(f"⚠️ Audio forward warning: {e}")

            async def receive_from_sarvam():
                nonlocal stt_ms, latest_partial
                try:
                    async for msg in sarvam_ws:
                        if msg.event == "transcript.partial":
                            part_text = (msg.text or "").strip()
                            if part_text:
                                latest_partial = part_text
                                await websocket.send_json({
                                    "type": "partial",
                                    "text": part_text,
                                    "language": target_lang
                                })
                        elif msg.event == "transcript.final":
                            final_t = (msg.text or "").strip()
                            if final_t:
                                received_final_text.append(final_t)
                                t_end = speech_end["t"] or speech_end["last_audio"]
                                stt_ms = max(0.0, (time.perf_counter() - t_end) * 1000)
                                await websocket.send_json({
                                    "type": "final",
                                    "text": final_t,
                                    "stt_ms": stt_ms,
                                    "language": target_lang
                                })
                            break
                        elif msg.event == "error":
                            err_msg = getattr(msg, "message", "Sarvam STT streaming error")
                            print(f"⚠️ Sarvam STT event warning: {err_msg}")
                            if getattr(msg, "is_fatal", False):
                                break
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"⚠️ Sarvam receive warning: {e}")

            task_send = asyncio.create_task(forward_audio_to_sarvam())
            task_recv = asyncio.create_task(receive_from_sarvam())

            # Wait for audio sending to complete (client sent 'stop' or disconnected)
            await task_send

            # Give Sarvam up to 3.0s to deliver the final transcription
            if not task_recv.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task_recv), timeout=3.0)
                except asyncio.TimeoutError:
                    print("⏱️ Sarvam final transcript timed out after 3.0s, falling back to latest partial...")
                except Exception as e:
                    print(f"⚠️ Sarvam wait warning: {e}")

            # Cancel receiver if still active
            if not task_recv.done():
                task_recv.cancel()
                try:
                    await task_recv
                except (asyncio.CancelledError, Exception):
                    pass

    except WebSocketDisconnect:
        return
    except Exception as e:
        print(f"⚠️ Live Sarvam WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        return

    # Fallback to latest partial if final was omitted
    if not received_final_text and latest_partial:
        received_final_text.append(latest_partial)
        t_end = speech_end["t"] or speech_end["last_audio"]
        stt_ms = max(0.0, (time.perf_counter() - t_end) * 1000)
        try:
            await websocket.send_json({
                "type": "final",
                "text": latest_partial,
                "stt_ms": stt_ms,
                "language": target_lang
            })
        except Exception:
            return

    if not received_final_text:
        try:
            await websocket.send_json({
                "type": "error",
                "message": "No speech detected. Please speak into the microphone."
            })
            await websocket.close()
        except Exception:
            pass
        return

    # Stream the RAG answer token by token
    full_query = " ".join(received_final_text).strip()
    active_lang = router_instance.resolve_language(query=full_query, requested_lang=target_lang, auto_detect=True)
    active_pipeline = router_instance.get_pipeline(active_lang)

    stream = handle_stream_query(
        active_pipeline,
        full_query,
        active_lang,
        5,
        evaluate and active_pipeline.evaluator is not None,
        None,
        stt_ms=stt_ms,
        mode="live_voice",
        use_reranker=use_reranker,
    )
    try:
        final_answer_text = None
        async for item in iterate_in_threadpool(stream):
            if item.get("type") == "answer":
                final_answer_text = item.get("answer")
            await websocket.send_json(item)

        voice_service = getattr(websocket.app.state, "voice_service", None)
        if voice_service and final_answer_text:
            audio_bytes, tts_ms = await run_in_threadpool(
                voice_service.generate_tts_audio,
                final_answer_text,
                active_lang,
                "shubh",
                1.0,
                22050,
                "bulbul:v3"
            )
            if audio_bytes:
                await websocket.send_json({
                    "type": "tts",
                    "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                    "tts_ms": round(tts_ms, 2)
                })
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        print(f"⚠️ Live voice answer error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

    try:
        await websocket.close()
    except Exception:
        pass
