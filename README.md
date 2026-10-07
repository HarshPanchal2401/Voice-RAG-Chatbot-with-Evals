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

# Voice quota (protects the Sarvam AI budget). Users = browser id + IP.
RAG_VOICE_LIMIT_PER_USER=5          # total voice uses per user (0 = unlimited)
RAG_VOICE_LIMIT_WINDOW_HOURS=0       # 0 = all-time limit (never resets); e.g. 24 = daily
RAG_VOICE_GLOBAL_DAILY_LIMIT=0       # all users together per day (0 = off)
RAG_TRUST_PROXY=0                    # 1 behind Hugging Face / Cloudflare to read the real visitor IP

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

## Deploy to Hugging Face Spaces (free)

The app runs on a free **Gradio + ZeroGPU** Space (Docker Spaces need a PRO plan). `space_app.py` downloads the indexes from a private dataset, reports ZeroGPU startup and serves the FastAPI app + web UI on port 7860. Everything runs on CPU.

Replace `harshpanchal241` with your Hugging Face username. Run the commands from the `v3` folder in PowerShell.

### 1. Log in (token with **Write** access, from your own account)
```powershell
hf auth login
hf auth whoami          # must print your username
```

### 2. Upload the indexes to a private dataset (one time, ~3.2 GB)
```powershell
hf repos create harshpanchal241/voice-rag-indexes --repo-type dataset --private
hf upload harshpanchal241/voice-rag-indexes voice_rag_builder . --repo-type dataset --include "*/bge_m3_*" --include "*/passage_metadata.sqlite" --include "*/pipeline_manifest.json"
```

### 3. Create the Space
On huggingface.co: **New → Space** → SDK **Gradio** → template **Blank** → hardware **ZeroGPU (Free)** → **Public**. Name it `voice-rag`.

### 4. Add secrets (Space → Settings → Variables and secrets → New secret)
| Name | Value |
| :--- | :--- |
| `GROQ_API_KEY` | Groq key |
| `SARVAM_API_KEY` | Sarvam key |
| `RAG_API_KEY` | app key users enter in the UI |
| `HF_TOKEN` | token **from the same account that owns the dataset** |
| `RAG_INDEX_REPO` | `harshpanchal241/voice-rag-indexes` (variable is fine) |

Optional variables: `RAG_VOICE_LIMIT_PER_USER` (default `5`, all-time), `RAG_VOICE_LIMIT_WINDOW_HOURS` (`0` = never resets), `RAG_VOICE_GLOBAL_DAILY_LIMIT`, `LANGSMITH_TRACING` / `LANGSMITH_API_KEY`.

### 5. Upload the code
```powershell
python scripts/upload_space.py --dry-run    # list the files that will be sent
python scripts/upload_space.py              # upload (never sends .env or the indexes)
```
Use this script instead of `hf upload --exclude "*..."`: on Windows the shell expands the `*` patterns and the command fails.

### 6. Check it
Open the Space → **Logs**. A healthy start shows:
```
HF_TOKEN belongs to 'harshpanchal241'
✅ [fetch_indexes] Done.
✅ [space_app] ZeroGPU startup reported.
✅ [LanguageRouter] Registered language: Gujarati (GU)
INFO:     Uvicorn running on http://0.0.0.0:7860
```
App URL: `https://harshpanchal241-voice-rag.hf.space`

### Update after code changes
```powershell
python scripts/upload_space.py
```

### Troubleshooting
| Log message | Fix |
| :--- | :--- |
| `RAG_INDEX_REPO is not set` | Add the `RAG_INDEX_REPO` variable. |
| `404 ... voice-rag-indexes` | `HF_TOKEN` is missing or from another account; use a token from the dataset owner (or make the dataset public). |
| `No @spaces.GPU function detected` | Upload the latest `space_app.py` (it reports ZeroGPU startup). |
| `402 Payment Required` | You picked Docker; create the Space with the **Gradio** SDK. |
| `403 ... namespace` | Wrong username in the command; check `hf auth whoami`. |

Notes: the free Space sleeps after 48 h without visitors. Voice-use counts survive restarts but reset when the Space is rebuilt (new upload).

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
