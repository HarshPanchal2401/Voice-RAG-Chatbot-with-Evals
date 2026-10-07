---
title: Voice RAG Chatbot
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: space_app.py
pinned: false
---

# Voice & Text Multilingual RAG System (v3)

A low-latency conversational Retrieval-Augmented Generation (RAG) system for **Gujarati** and **Hindi**, with voice input and spoken replies. Questions typed in English are answered in English from the Gujarati/Hindi index.

- **Hybrid search**: dense multilingual embeddings (`BAAI/bge-m3`) in a FAISS HNSW index plus BGE-M3 sparse lexical weights, fused with Reciprocal Rank Fusion (RRF).
- **Re-ranking (optional)**: a fast LLM listwise re-ranker with per-passage relevance grades (default), a local multilingual cross-encoder (`BAAI/bge-reranker-v2-m3`, opt-in, needs ~1–2 GB extra RAM), or a lexical fallback. Low-relevance passages are dropped, and the system says so when nothing relevant is found.
- **Generation**: Groq-hosted open models with inline citations (`[સંદર્ભ 1]`, `[संदर्भ 1]`, `[Source 1]`) and conversation memory (follow-up questions are rewritten into standalone questions before retrieval).
- **Voice**: Sarvam AI `saaras:v3` speech-to-text with Groq Whisper fallback, language auto-detection, and Sarvam `bulbul:v3` text-to-speech streamed **sentence by sentence** while the answer is generated.
- **Streaming**: Server-Sent Events for text, and a WebSocket for live microphone audio.
- **Evaluation**: offline DeepEval/LangSmith harness and an optional online evaluator. See [Evaluation](#evaluation).
- **Security**: optional API key, per-IP rate limiting, CORS allow-list, upload size and live-session limits.

---

## Getting started

### Prerequisites
- Python 3.10+ (developed on 3.14 for Windows and 3.11 in Docker).
- About 6 GB of free RAM for both languages with the default indexes. Building the compressed index saves about 1.3 GB (see [Memory](#memory)).
- API keys: [Groq](https://console.groq.com) (required), [Sarvam AI](https://www.sarvam.ai) (voice), [LangSmith](https://smith.langchain.com) (optional).
- The prebuilt indexes in `voice_rag_builder/<language>/` (`bge_m3_dense_hnsw*.index`, `bge_m3_sparse_lexical.npz`, `passage_metadata.sqlite`, `pipeline_manifest.json`).

### Install
```bash
cd v3
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
pip install -r requirements-dev.txt   # tests + CLI microphone support
```

### Configure (`v3/.env`)
```env
# Required
GROQ_API_KEY=...
# Voice (speech-to-text and text-to-speech)
SARVAM_API_KEY=...
# Optional: Hugging Face token for model downloads
HF_TOKEN=...

# Models (defaults shown). The judge must differ from the generator and the re-ranker.
RAG_LLM_MODEL=qwen/qwen3.8-27b
RAG_JUDGE_MODEL=openai/gpt-oss-120b
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_RERANKER_BACKEND=llm            # llm | cross_encoder | lexical

# Security (recommended whenever the server is reachable by others)
RAG_API_KEY=choose-a-long-random-string
RAG_CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
RAG_RATE_LIMIT_PER_MIN=30
RAG_MAX_UPLOAD_MB=10
RAG_WS_MAX_SECONDS=60

# Observability (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT="Voice RAG System"
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```
Further tuning variables (`RAG_FINAL_TOP_K`, `RAG_HISTORY_TURNS`, `RAG_CONDENSE_QUERY`, `RAG_RERANK_MIN_SCORE`, `RAG_MIN_DENSE_SCORE`, `RAG_MAX_QUERY_LEN`, `RAG_TTS_MAX_CHARS`, `RAG_LANGUAGES`, …) are documented in `core/config.py` and `core/security.py`.

When `RAG_API_KEY` is set, every `/api/*` request needs the header `X-API-Key`. The web UI asks for the key once and remembers it in the browser.

---

## Running

```bash
# API + web dashboard
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
- Dashboard: http://localhost:8000
- API docs: http://localhost:8000/docs

The microphone only works on `localhost` or over HTTPS, because browsers block it on plain-HTTP network addresses.

```bash
# Interactive CLI chatbot (text and microphone)
python chatbot.py

# Latency benchmark (p50/p95 per stage, both languages)
python measure_latency.py --help
```

### Docker
```bash
docker build -t voice-rag:v3 .
docker run --env-file .env -p 8000:8000 \
  -v hf-cache:/home/app/.cache/huggingface \
  voice-rag:v3
```
The image installs CPU-only PyTorch and never contains your `.env`. Pass the keys at run time. The `voice_rag_builder/` indexes must be present in the build context. Mount the Hugging Face cache as a volume so BGE-M3 is not downloaded on every start.

### Memory
Each language's full-precision HNSW index takes about 1.3 GB of RAM. Convert it once to an fp16 index of about half the size. The server picks it up automatically, and the script deletes its output if search quality drops:
```bash
python scripts/build_fp16_index.py --lang gujarati
python scripts/build_fp16_index.py --lang hindi
```

---

## API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Health, loaded languages, models, whether an API key is required |
| `/api/v1/languages` | GET | Loaded languages and index sizes |
| `/api/v1/sample-queries?lang=gu&count=20` | GET | Sample questions from the golden dataset |
| `/api/v1/query/text` | POST | Text question, full JSON answer with sources and latency breakdown |
| `/api/v1/query/text/stream` | POST | Same, streamed as SSE: `token`, `tts`, `meta`, `tts_done`, `evaluation`, `error`, `done` |
| `/api/v1/query/voice` | POST | Audio file upload (wav, mp3, m4a, ogg, webm, flac, mp4) to speech-to-text, answer and spoken reply |
| `/api/v1/voice/live` | WebSocket | Live 16 kHz PCM microphone stream with partial transcripts and a streamed answer |
| `/api/v1/voice/tts` | POST | Text-to-speech for any text (gu, hi, en) |
| `/api/v1/retrieve` | POST | Retrieval only, with `sort_by` = `rrf`, `dense`, `sparse` or `rerank` |
| `/api/v1/evaluate` | POST | Evaluate a question, answer and contexts |
| `/api/v1/feedback` | POST | Thumbs up/down on an answer (logged to LangSmith) |

Text query body:
```json
{
  "query": "મેનહટન પ્રોજેક્ટ શું હતો?",
  "language": "auto",
  "top_k": 5,
  "use_reranker": false,
  "sort_by": "rrf",
  "auto_detect_language": true,
  "evaluate": false,
  "voice_reply": false,
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

---

## Testing

```bash
python -m pytest -q                 # unit tests (no network, no models)
python -m pytest -q -m integration  # needs the indexes, models and API keys
```

## Evaluation

See [`evaluation/README.md`](evaluation/README.md) for the full guide.
