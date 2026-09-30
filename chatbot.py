"""
Gujarati Voice & Text RAG Chatbot
=================================

Features:
- 🚀 In-Memory Model Persistence: Loads BGE-M3 + FAISS indices ONCE at startup.
- ⚡ Instant Query Response: 0s reload time, ~400ms retrieval, ~1s total answer.
- 💬 Interactive Text Chat: Type any question in Gujarati or English.
- 🌊 Live Token Streaming: Real-time LLM typing effect.
- 🎤 Future-Ready Voice Input: Type `/voice` to speak directly through microphone (Sarvam AI STT).
- 📜 Source Inspection: View retrieved references and exact latency breakdown.

Usage:
  python chatbot.py
"""

import os
import sys
import time
import asyncio
from dotenv import load_dotenv

load_dotenv()

# UTF-8 encoding support for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline import GujaratiHybridRAGNoReranker


# ============================================================
# Optional Live Voice STT via Sarvam AI
# ============================================================

async def record_and_transcribe_voice(language_code="gu-IN"):
    """
    Captures live audio from microphone and returns transcribed text using Sarvam AI.
    """
    import base64
    from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd
    import sounddevice as sd

    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        print("\n❌ Error: SARVAM_API_KEY is not set in .env file.")
        return None

    client = AsyncSarvamAI(api_subscription_key=api_key)
    stop_event = asyncio.Event()
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    final_text = []

    def audio_cb(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    print("\n🎤 [Microphone Active] Speak your question now in Gujarati...")
    print("⏳ Listening (stops automatically when you finish speaking)...\n")

    sample_rate = 16000
    chunk_samples = int(sample_rate * 0.1)  # 100ms

    try:
        async with client.speech_to_text_realtime_streaming.connect(
            language_code=language_code,
            stream_type="fast",
        ) as ws:

            async def send_audio():
                with sd.RawInputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=chunk_samples,
                    callback=audio_cb,
                ):
                    while not stop_event.is_set():
                        try:
                            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.15)
                            b64 = base64.b64encode(chunk).decode("utf-8")
                            await ws.send_realtime_audio_input(RealtimeAudioInput(audio=b64))
                        except asyncio.TimeoutError:
                            continue
                    try:
                        await ws.send_realtime_end(RealtimeEnd())
                    except Exception:
                        pass

            async def receive_events():
                async for msg in ws:
                    if msg.event == "transcript.partial":
                        print(f"\r🎙️ [Speaking]: {msg.text}", end="", flush=True)
                    elif msg.event == "transcript.final":
                        t = msg.text.strip()
                        if t:
                            print(f"\r✅ [Heard]: {t}\n")
                            final_text.append(t)
                            stop_event.set()
                            return
                    elif msg.event == "error":
                        if msg.is_fatal:
                            stop_event.set()
                            return

            send_t = asyncio.create_task(send_audio())
            recv_t = asyncio.create_task(receive_events())
            await asyncio.gather(send_t, recv_t)

    except Exception as e:
        print(f"\n⚠️ Voice capture error: {e}")

    return " ".join(final_text).strip() if final_text else None


# ============================================================
# Chatbot Interface
# ============================================================

def print_banner():
    banner = """
===============================================================
       🤖 GUJARATI VOICE & TEXT RAG ASSISTANT + EVAL
===============================================================
  Model     : BGE-M3 (In-Memory ⚡) + Groq (GPT-OSS-20B)
  Commands  :
    /voice    -> Speak your question via microphone 🎙️
    /eval     -> Toggle Golden Dataset DeepEval evaluation 🎯
    /sources  -> Toggle showing retrieved passage sources 📑
    /stream   -> Toggle streaming response on/off 🌊
    /clear    -> Clear screen 🧹
    /exit     -> Quit chatbot 👋
===============================================================
"""
    print(banner)


def format_evaluation_scorecard(eval_res):
    """Formats DeepEval evaluation metrics into a clean terminal table."""
    if not eval_res:
        return ""

    if not eval_res.get("is_golden"):
        return f"\n⚠️  \033[93m{eval_res.get('warning', 'Query not found in Golden Dataset.')}\033[0m\n"

    scores = eval_res.get("scores", {})
    gt_ans = eval_res.get("ground_truth_answer", "N/A")
    q_type = eval_res.get("query_type", "UNKNOWN")
    qid = eval_res.get("query_id", "N/A")
    explanation = scores.get("explanation", "")

    score_card = f"""
┌─────────────────────────────────────────────────────────────┐
│ 🎯 DEEPEVAL EVALUATION SCORECARD (Golden Query #{qid} | {q_type})
├─────────────────────────────────────────────────────────────┤
│ • Overall Score       : {scores.get('overall_score', 0):.2f} / 1.00
│ • Answer Correctness  : {scores.get('answer_correctness', 0):.2f} / 1.00
│ • Semantic Similarity : {scores.get('answer_similarity', 0):.2f} / 1.00
│ • Faithfulness        : {scores.get('faithfulness', 0):.2f} / 1.00
│ • Answer Relevance    : {scores.get('answer_relevance', 0):.2f} / 1.00
│ • Context Recall      : {scores.get('context_recall', 0):.2f} / 1.00
│ • Top-K Hit Rate      : {scores.get('hit_rate_at_k', 0):.1f}
├─────────────────────────────────────────────────────────────┤
│ 📖 Ground Truth Answer:
│   "{gt_ans}"
"""
    if explanation:
        score_card += f"""├─────────────────────────────────────────────────────────────┤
│ 💡 Judge Note: {explanation}
"""
    score_card += "└─────────────────────────────────────────────────────────────┘\n"
    return score_card


def main():
    print_banner()
    print("⏳ Initializing RAG Pipeline into RAM (Hold on ~5s)...")
    t0 = time.time()
    
    # ── LOAD MODEL ONCE INTO MEMORY ─────────────────────────────
    pipeline = GujaratiHybridRAGNoReranker(load_groq=True, verbose=True)
    load_time = time.time() - t0
    print(f"\n🚀 Pipeline ready in memory! (Initial load: {load_time:.2f}s)")
    print("👉 Type your Gujarati question below or use '/voice' to speak.\n")

    show_sources = True
    streaming_mode = True
    eval_mode = False

    while True:
        try:
            user_input = input("\n👤 You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting. Good bye!")
            break

        if not user_input:
            continue

        # Command handling
        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit", "q"):
            print("👋 Exiting Chatbot. Have a nice day!")
            break

        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        elif cmd == "/sources":
            show_sources = not show_sources
            print(f"📑 Show Sources: {'ENABLED' if show_sources else 'DISABLED'}")
            continue

        elif cmd == "/eval":
            eval_mode = not eval_mode
            print(f"🎯 Golden Dataset Evaluation: {'ENABLED' if eval_mode else 'DISABLED'}")
            continue

        elif cmd == "/stream":
            streaming_mode = not streaming_mode
            print(f"🌊 Streaming Mode: {'ENABLED' if streaming_mode else 'DISABLED'}")
            continue

        elif cmd == "/help":
            print_banner()
            continue

        elif cmd == "/voice":
            print("Preparing microphone...")
            try:
                voice_text = asyncio.run(record_and_transcribe_voice())
                if voice_text:
                    user_input = voice_text
                else:
                    print("⚠️ No speech recognized. Please try again or type.")
                    continue
            except Exception as e:
                print(f"⚠️ Microphone error: {e}")
                continue

        # ── PROCESS QUERY IN-MEMORY ─────────────────────────────────
        print("\n🤖 Assistant > ", end="", flush=True)

        full_answer = ""
        retrieved_docs = []

        if streaming_mode:
            meta = {}
            for chunk in pipeline.ask_stream(user_input):
                if chunk["type"] == "token":
                    full_answer += chunk["content"]
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "meta":
                    meta = chunk

            print("\n")
            
            # Latency Badge
            if meta:
                retrieved_docs = meta.get("documents", [])
                ret_ms = meta["retrieval_timings"]["retrieval_total_ms"]
                ttft_ms = meta["ttft_ms"]
                total_ms = meta["total_ms"]
                print(f"⚡ [Latency: Retrieval={ret_ms:.0f}ms | TTFT={ttft_ms:.0f}ms | Total={total_ms:.0f}ms]")

                if show_sources and retrieved_docs:
                    print("\n--- 📑 Top Retrieved References ---")
                    for r, doc in enumerate(retrieved_docs, start=1):
                        print(f"[{r}] (RRF={doc.get('rrf_score', 0):.4f}) {doc['text'][:150]}...")
        else:
            result = pipeline.ask(user_input)
            full_answer = result["answer"]
            retrieved_docs = result.get("documents", [])
            print(full_answer)
            print(f"\n⚡ [Latency: Retrieval={result['retrieval_timings']['retrieval_total_ms']:.0f}ms | Total={result['total_ms']:.0f}ms]")

            if show_sources and retrieved_docs:
                print("\n--- 📑 Top Retrieved References ---")
                for r, doc in enumerate(retrieved_docs, start=1):
                    print(f"[{r}] (RRF={doc.get('rrf_score', 0):.4f}) {doc['text'][:150]}...")

        # ── DEEPEVAL EVALUATION AGAINST GOLDEN DATASET ──────────────
        if eval_mode:
            print("\n⏳ Evaluating with DeepEval against Golden Dataset...", flush=True)
            eval_res = pipeline.evaluate(user_input, full_answer, retrieved_docs)
            print(format_evaluation_scorecard(eval_res))


if __name__ == "__main__":
    main()
