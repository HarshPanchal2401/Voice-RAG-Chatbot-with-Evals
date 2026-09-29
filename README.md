# 🎙️ Voice & Text Multilingual RAG System with DeepEval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange.svg)](https://groq.com)
[![Sarvam AI](https://img.shields.io/badge/STT-Sarvam%20Saaras%3Av3-purple.svg)](https://sarvam.ai)
[![DeepEval](https://img.shields.io/badge/Evaluation-DeepEval-success.svg)](https://github.com/confident-ai/deepeval)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-black.svg)](https://smith.langchain.com)

A production-grade, low-latency conversational Retrieval-Augmented Generation (RAG) engine designed for Indic languages (**Gujarati** and **Hindi**) and English. It pairs dense multilingual embeddings and sparse lexical search with Reciprocal Rank Fusion (RRF), secondary listwise re-ranking, real-time speech-to-text (STT), SSE token streaming, and automated evaluation using DeepEval and LangSmith.

---

## 🌟 Key Highlights

- **⚡ Low-Latency Hybrid Search**:
  - **Dense**: Multilingual `BAAI/bge-m3` embeddings indexed with FAISS (HNSW / Flat).
  - **Sparse**: Lexical sparse representations via SciPy Compressed Sparse Column (CSC) matrices.
  - **Fusion & Deterministic Sorting**: Reciprocal Rank Fusion (RRF) with multi-criteria sorting (`rrf`, `dense`, `sparse`, `rerank`).
- **🎯 Secondary Listwise Re-ranking**:
  - LLM judge and cross-scoring re-ranker to maximize Precision@K and MRR for complex domain questions.
- **🗣️ Indic Voice Integration**:
  - Dual Speech-to-Text (STT) pipeline: **Sarvam AI Saaras:v3** (primary for Indic accents) with automatic fallback to **Groq Whisper Large v3**.
  - Real-time microphone audio streaming over WebSockets and multipart file uploads.
- **🧠 Smart Script Auto-Detection**:
  - Zero-latency Unicode block script analyzer detecting Gujarati (`\u0A80-\u0AFF`) and Hindi / Devanagari (`\u0900-\u097F`) for automatic pipeline routing.
- **🌊 Real-time Token Streaming**:
  - Server-Sent Events (SSE) `/stream` and WebSocket endpoints with millisecond-level Time-To-First-Token (TTFT) metrics.
- **📊 Decoupled RAG Triad Evaluation**:
  - Asynchronous, non-blocking evaluation with **DeepEval**: Faithfulness, Answer Relevancy, Contextual Recall, and Hallucination detection.
  - End-to-end distributed tracing and token analytics via **LangSmith**.
- **💻 Dual Interfaces**:
  - Glassmorphic Web Dashboard (`static/index.html`) with live microphone recording, sample queries, and latency breakdowns.
  - Interactive **Streamlit** Chatbot (`chatbot.py`).

---

## 🏗️ System Architecture

```text
User Input (Audio / Text)
       │
       ▼
 [Voice STT Service] ──► (Sarvam AI Saaras:v3 / Groq Whisper)
       │
       ▼ Text Query
 [Script Detector & Language Router] ──► Routes to Gujarati / Hindi Pipeline
       │
       ├─────────────────────────────────┐
       ▼ (Dense Search)                  ▼ (Sparse Lexical Search)
 [FAISS HNSW Vector Store]         [SciPy Compressed Sparse Matrix]
       │                                 │
       └───────────────┬─────────────────┘
                       ▼
         [Reciprocal Rank Fusion (RRF)]
                       │
                       ▼
      [Secondary Listwise LLM Reranker]
                       │
                       ▼
           [SQLite Metadata Store] ──► Retrieve passage chunk text
                       │
                       ▼
    [Groq Llama 3.3 70B Versatile LLM]
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
 [SSE / WebSocket Stream]      [Async Background DeepEval]
 (Live Tokens to Client)       (Faithfulness & Relevancy)
                                       │
                                       ▼
                              [LangSmith Tracing]
```

---

## 📂 Project Structure

```text
v3/
├── app.py                          # Lean FastAPI application & lifespan management
├── chatbot.py                      # Interactive Streamlit voice & text chatbot
├── measure_latency.py              # Performance profiling & benchmarking utility
├── requirements.txt                # Python package dependencies
├── ARCHITECTURE.md                 # In-depth architectural & pipeline specifications
│
├── core/                           # System core & global settings
│   ├── config.py                   # Central settings, environment variables, hyperparameters
│   └── tracing.py                  # LangSmith tracing, LLM wrapper, trace flusher
│
├── pipeline/                       # Modular RAG pipeline components
│   ├── embeddings.py               # Native BAAI/bge-m3 query encoder
│   ├── metadata_store.py           # SQLite passage & chunk metadata store
│   ├── retriever.py                # HybridRetriever: FAISS + Sparse CSC + RRF + Multi-mode sorting
│   ├── reranker.py                 # RAGReranker: Multilingual Listwise LLM Re-ranker
│   ├── generator.py                # Prompt templates, script detection, and Groq streaming
│   ├── single_pipeline.py          # SingleLanguagePipeline coordinating one language index
│   └── router.py                   # LanguageRouter with smart script auto-detection
│
├── services/                       # Application services
│   ├── voice_service.py            # STT service (Sarvam AI Saaras:v3 + Groq Whisper)
│   └── evaluation_service.py       # DeepEval evaluation harness & Groq judge model
│
├── schemas/                        # Type-safe Pydantic data contracts
│   ├── requests.py                 # TextQueryRequest, RetrieveRequest, EvaluateRequest
│   ├── responses.py                # RAGResponse, RetrievalResponse, HealthResponse
│   └── models.py                   # SourceDocument, LatencyBreakdown, RetrievalTimings, EvaluationResult
│
├── api/                            # Modular FastAPI route controllers
│   ├── dependencies.py             # Dependency injection (router, voice service)
│   ├── helpers.py                  # Latency calculations and async evaluation handlers
│   ├── routes_system.py            # Health check, language list, sample queries, UI static mount
│   ├── routes_query.py             # Synchronous text and voice query endpoints
│   ├── routes_stream.py            # SSE token streaming & live WebSocket STT
│   ├── routes_retrieval.py         # Search-only endpoint
│   └── routes_evaluation.py        # Standalone DeepEval evaluation endpoint
│
├── evaluation/                     # Offline evaluation framework & metrics
│   ├── client.py                   # Client adapter for offline evaluation
│   ├── data_loader.py              # Golden dataset loader
│   ├── pipeline_evaluator.py       # Multi-query pipeline evaluation runner
│   ├── report_generator.py         # Markdown and JSON report generator
│   ├── run_evaluation.py           # Evaluation entry point script
│   ├── components/                 # Component evaluators (Retriever, Generator, Application)
│   └── metrics/                    # Metric computation (Faithfulness, Relevancy, Hallucination)
│
├── static/                         # Web frontend dashboard
│   ├── index.html                  # Responsive modern UI
│   ├── style.css                   # Glassmorphic dark styling
│   ├── app.js                      # Client controller, audio recording, SSE consumer
│   ├── golden_sample_queries.json  # Gujarati sample questions
│   └── golden_sample_queries_hi.json# Hindi sample questions
│
├── tests/                          # Automated test suite
│   ├── test_pipeline_sorting.py    # Unit tests for RRF sorting & script detection
│   ├── test_api.py                 # End-to-end FastAPI endpoint integration tests
│   └── test_multilingual_router.py # Language router and golden query verification
│
├── Data/                           # Datasets & Golden corpora (gitignored)
└── voice_rag_builder/              # Prebuilt FAISS indexes & SQLite databases (gitignored)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.14)
- Git
- Valid API keys:
  - [Groq API Key](https://console.groq.com)
  - [Sarvam AI API Key](https://www.sarvam.ai) *(for Indic voice recognition)*
  - [LangSmith API Key](https://smith.langchain.com) *(optional, for observability)*

### 2. Clone the Repository
```bash
git clone https://github.com/HarshPanchal2401/Voice-RAG-Chatbot-with-Evals.git
cd Voice-RAG-Chatbot-with-Evals/v3
```

### 3. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the `v3/` directory:
```env
# LLM & STT
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# LangSmith Observability (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=voice-rag-v3

# DeepEval (Optional)
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

---

## 🖥️ Running the Application

### 1. Launch FastAPI Backend & Web Dashboard
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 2. Launch Streamlit Chatbot
```bash
streamlit run chatbot.py
```

### 3. Run Performance & Latency Benchmark
```bash
python measure_latency.py
```

---

## 🧪 Testing & Evaluation

### Run Unit & Integration Tests
```bash
# Unit test for RRF ranking, multi-mode sorting, and script detection
python -u tests/test_pipeline_sorting.py

# Full API integration test across all 7 endpoints
python -u tests/test_api.py

# Multilingual router verification
python -u tests/test_multilingual_router.py
```

### Run DeepEval Offline Evaluation
```bash
python evaluation/run_evaluation.py --component all --language gujarati
```
Evaluation reports will be generated in `evaluation/reports/`.

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/query` | `POST` | Synchronous RAG text query with detailed latency breakdown |
| `/query/voice` | `POST` | Audio file upload (`.wav`, `.mp3`) for STT transcription + RAG answer |
| `/stream` | `POST` | Server-Sent Events (SSE) token stream with post-stream metrics |
| `/ws/voice` | `WebSocket`| Real-time live audio chunk STT and query processing |
| `/retrieve` | `POST` | Search-only retrieval with custom `sort_by` (`rrf`, `dense`, `sparse`, `rerank`) |
| `/evaluate` | `POST` | On-demand DeepEval evaluation for any question-answer-context pair |
| `/health` | `GET` | Health check and loaded language router status |
| `/languages` | `GET` | Supported languages and index statistics |
| `/sample-queries`| `GET` | Golden test queries for Gujarati and Hindi |

---

## 🔐 GitHub Authentication & Push Troubleshooting

If you encounter:
```text
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/...'
```

GitHub requires a **Personal Access Token (PAT)** or the **GitHub CLI** instead of your account password:

### Method A: Using GitHub CLI (`gh`) (Recommended)
```bash
# 1. Authenticate with GitHub CLI
gh auth login

# 2. Select GitHub.com -> HTTPS -> Login with a web browser
# 3. Push your branch
git push -u origin main
```

### Method B: Using a Personal Access Token (PAT)
1. Go to **GitHub** → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
2. Generate a token with the `repo` scope.
3. Update your Git credential cache:
   ```bash
   # On Windows PowerShell:
   cmdkey /delete:git:https://github.com
   git push -u origin main
   ```
4. When prompted for username, enter your GitHub username; for password, paste your **Personal Access Token (`ghp_...`)**.
