# Voice & Text RAG System (v3 Architecture & Pipeline Guide)

## 1. Overview
The **Multi-Language Voice & Text RAG System (v3)** is a production-grade, low-latency conversational RAG engine supporting **Gujarati** and **Hindi** with extensible multilingual routing. It combines dense semantic retrieval, sparse lexical retrieval, reciprocal rank fusion (RRF), secondary listwise re-ranking, and speech-to-text (STT) capabilities.

---

## 2. Refactored Project Structure

```
v3/
├── app.py                          # Lean FastAPI application entry point & lifespan
├── chatbot.py                      # Interactive terminal Voice & Text Chatbot
├── measure_latency.py              # Performance profiling & benchmarking utility
│
├── core/                           # System core & cross-cutting infrastructure
│   ├── __init__.py
│   ├── config.py                   # Central settings, environment variables, hyper-parameters
│   └── tracing.py                  # LangSmith tracing, LLM wrapping, async feedback logging
│
├── pipeline/                       # Modular RAG Pipeline stages
│   ├── __init__.py
│   ├── embeddings.py               # Native BGE-M3 Multilingual Query Encoder & Quantization
│   ├── retriever.py                # HybridRetriever: FAISS HNSW + SciPy CSC + RRF + Multi-mode Sorting
│   ├── reranker.py                 # RAGReranker: Multilingual Listwise LLM & Lexical Cross-Score Re-ranker
│   ├── metadata_store.py           # MetadataStore: SQLite passage/chunk metadata store & fetch
│   ├── generator.py                # Multilingual prompting, context construction, and Groq streaming
│   ├── single_pipeline.py          # SingleLanguagePipeline coordinating one language index
│   └── router.py                   # LanguageRouter with smart script auto-detection
│
├── services/                       # Application services
│   ├── __init__.py
│   ├── voice_service.py            # STT service (Sarvam AI Saaras:v3 + Groq Whisper)
│   └── evaluation_service.py       # DeepEval evaluation engine & Golden dataset evaluator
│
├── schemas/                        # Type-safe Pydantic data contracts
│   ├── __init__.py
│   ├── requests.py                 # TextQueryRequest, RetrieveRequest, EvaluateRequest
│   ├── responses.py                # RAGResponse, RetrievalResponse, HealthResponse
│   └── models.py                   # SourceDocument, LatencyBreakdown, RetrievalTimings, EvaluationResult
│
├── api/                            # Modular FastAPI endpoints
│   ├── __init__.py
│   ├── dependencies.py             # FastAPI dependency injection (get_router, get_voice_service)
│   ├── helpers.py                  # Response formatting, latency calculation, and async evaluation
│   ├── routes_system.py            # Health check, languages info, sample queries, UI serving
│   ├── routes_query.py             # Synchronous text and voice query endpoints
│   ├── routes_stream.py            # SSE token streaming and live microphone STT WebSocket
│   ├── routes_retrieval.py         # Search-only endpoint
│   └── routes_evaluation.py        # Standalone DeepEval evaluation endpoint
│
├── tests/                          # Automated test suite
│   ├── __init__.py
│   ├── test_pipeline_sorting.py    # Unit tests for RRF ordering, multi-mode sorting, and script detection
│   ├── test_api.py                 # Full FastAPI integration tests across all 7 endpoints
│   └── test_multilingual_router.py # Pipeline routing and golden evaluation tests
│
├── static/                         # Web dashboard assets (index.html, app.js, style.css)
├── Data/                           # Datasets & Golden datasets (Gujarati, Hindi)
├── voice_rag_builder/              # Prebuilt FAISS indexes, sparse matrices, SQLite DBs
└── evaluation/                     # Benchmark evaluation suite
```

---

## 3. The Sorted Pipeline Architecture

Previously, the pipeline lacked clear sorting controls, and re-ranking was hidden in an offline folder. The new `HybridRetriever` and `SingleLanguagePipeline` introduce deterministic multi-stage sorting:

```
User Query
    │
    ▼
1. Query Encoding (NativeBGEM3)
    ├── Dense Vector (1024-d, L2-normalized)
    └── Sparse Lexical Weights (Token IDs + Max-pooled weights)
    │
    ▼
2. Dual Search
    ├── FAISS HNSW Dense Search (top-50 candidates)
    └── SciPy CSC Sparse Lexical Search (top-50 candidates)
    │
    ▼
3. Reciprocal Rank Fusion (RRF) & Initial Sorting
    └── Score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
    └── Strictly sorted descending
    │
    ▼
4. SQLite Metadata Enrichment
    └── Fetches passage details while preserving strict rank order
    │
    ▼
5. Optional Listwise Re-Ranking (RAGReranker)
    ├── Groq Indic/multilingual LLM (Qwen 2.5 27B / GPT-OSS) listwise scoring
    └── Deterministic lexical cross-scoring fallback if offline
    │
    ▼
6. Flexible Document Sorting (`sort_by`)
    ├── 'rrf'    -> Ranked by combined hybrid score (default)
    ├── 'dense'  -> Ranked strictly by dense semantic similarity
    ├── 'sparse' -> Ranked strictly by lexical token score
    └── 'rerank' -> Ranked strictly by secondary re-ranker score
```

---

## 4. System Flow Improvements

### A. Intelligent Script Auto-Detection
- Queries no longer default blindly to the wrong language index.
- If `language="auto"` or `auto_detect_language=True`:
  - Characters in Gujarati Unicode block `\u0A80-\u0AFF` automatically route to **Gujarati**.
  - Characters in Devanagari Unicode block `\u0900-\u097F` automatically route to **Hindi**.

### B. Audio & Voice Query Flow
- Audio uploaded to `/api/v1/query/voice` or streamed via WebSocket `/api/v1/voice/live`.
- Transcribed using **Sarvam AI (saaras:v3)** or **Groq Whisper (whisper-large-v3)**.
- Transcribed text undergoes script detection to ensure the right vector index is searched even if the user selected a different language tab.

### C. Decoupled Asynchronous Evaluation
- On streaming endpoints (`/api/v1/query/text/stream` and `/api/v1/voice/live`), generation tokens stream to the user in real-time (~300ms TTFT).
- Once generation finishes, DeepEval evaluation executes in the background and emits as an `evaluation` event. The user never waits on evaluation to read the answer.

### D. Comprehensive Latency Breakdown
Every response includes detailed stage-by-stage latencies:
- `stt_ms`: Speech-to-Text latency (if voice query)
- `retrieval.query_encoding_ms`: Dense + Sparse embedding generation
- `retrieval.dense_search_ms`: FAISS HNSW search
- `retrieval.sparse_search_ms`: SciPy CSC sparse inverted index search
- `retrieval.rrf_ms`: Reciprocal Rank Fusion & sorting
- `retrieval.metadata_ms`: SQLite metadata lookup
- `retrieval.rerank_ms`: Re-ranking latency (if enabled)
- `retrieval.retrieval_total_ms`: Total retrieval latency
- `ttft_ms`: Time to first token (streaming)
- `llm_ms`: LLM answer generation time
- `eval_ms`: Post-generation DeepEval evaluation time
- `total_ms`: Total end-to-end user-perceived latency

---

## 5. Clean Modular Architecture
All core functionalities are neatly organized into dedicated packages:
- `core/`: Global settings (`config.py`) and LangSmith telemetry (`tracing.py`).
- `pipeline/`: Embeddings (`NativeBGEM3`), metadata lookup (`MetadataStore`), hybrid retrieval with deterministic sorting (`HybridRetriever`), reranker (`RAGReranker`), LLM generator (`generator.py`), single language pipeline (`SingleLanguagePipeline`), and multilingual routing (`LanguageRouter`).
- `services/`: Voice STT (`VoiceService`) and DeepEval evaluation (`DeepEvalEvaluator`).
- `schemas/`: Pydantic request, response, and latency data models.
- `api/`: Modular FastAPI route handlers.
- `evaluation/`: Offline evaluation harness with metrics, loader, and report generators.
- `tests/`: Automated unit and integration test suite.
