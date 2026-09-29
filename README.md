# 🎙️ Voice & Text Multilingual RAG System with DeepEval

A production-ready, low-latency conversational Retrieval-Augmented Generation (RAG) system for Indic languages (**Gujarati** and **Hindi**) and English. Features hybrid retrieval (dense BGE-M3 + sparse lexical search), Reciprocal Rank Fusion (RRF), secondary listwise re-ranking, open-source (OSS) LLM generation via Groq, real-time speech-to-text (STT), SSE token streaming, and automated evaluation using DeepEval.

---

## 🌟 Features

- **Hybrid Search**: Dense multilingual embeddings (`BAAI/bge-m3`) with FAISS + lexical sparse representations (SciPy CSC).
- **Rank Fusion & Sorting**: Reciprocal Rank Fusion (RRF) with multi-criteria deterministic sorting (`rrf`, `dense`, `sparse`, `rerank`).
- **Listwise Re-ranking**: Open-source LLM judge (`openai/gpt-oss-120b`) for precision re-ranking.
- **Indic Voice Recognition**: Dual STT pipeline using **Sarvam AI Saaras:v3** with automatic fallback to **Groq Whisper Large v3**.
- **Smart Script Detection**: Unicode-based script analyzer detecting Gujarati (`\u0A80-\u0AFF`) and Devanagari/Hindi (`\u0900-\u097F`).
- **Real-Time Streaming**: Server-Sent Events (SSE) `/stream` and WebSocket endpoints with millisecond-level TTFT metrics.
- **Automated Evaluation**: Asynchronous evaluation using **DeepEval** (Faithfulness, Answer Relevancy, Recall) and LangSmith tracing.
- **Web & Chat Interfaces**: Glassmorphic web UI (`static/index.html`) and Streamlit chatbot (`chatbot.py`).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- API keys: [Groq](https://console.groq.com), [Sarvam AI](https://www.sarvam.ai) *(for Indic voice)*, [LangSmith](https://smith.langchain.com) *(optional)*

### 2. Installation
```bash
git clone https://github.com/HarshPanchal2401/Voice-RAG-Chatbot-with-Evals.git
cd Voice-RAG-Chatbot-with-Evals/v3

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the `v3/` root:
```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
HF_TOKEN=your_huggingface_token_here

# Open-Source (OSS) Models via Groq
RAG_LLM_MODEL=openai/gpt-oss-120b
RAG_RERANKER_MODEL=openai/gpt-oss-120b
RAG_JUDGE_MODEL=openai/gpt-oss-120b
RAG_EMBEDDING_MODEL=BAAI/bge-m3

# Observability (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT="Voice RAG System"

# DeepEval (Optional)
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

---

## 🖥️ Running the Project

### FastAPI Backend & Web Dashboard
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- Dashboard: [http://localhost:8000](http://localhost:8000)
- API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

### Streamlit Chatbot
```bash
streamlit run chatbot.py
```

### Latency Benchmarking
```bash
python measure_latency.py
```

---

## 🧪 Testing & Evaluation

```bash
# Unit tests (sorting & script detection)
python -u tests/test_pipeline_sorting.py

# Full API integration tests
python -u tests/test_api.py

# Offline DeepEval evaluation runner
python evaluation/run_evaluation.py --component all --language gujarati
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/query` | `POST` | Text query with full latency breakdown |
| `/query/voice` | `POST` | Audio file upload (`.wav`, `.mp3`) for STT + answer |
| `/stream` | `POST` | Server-Sent Events (SSE) token stream |
| `/ws/voice` | `WebSocket` | Live microphone audio chunk streaming |
| `/retrieve` | `POST` | Retrieval-only search with configurable sorting |
| `/evaluate` | `POST` | On-demand DeepEval evaluation for question/answer/context |
| `/health` | `GET` | Health status and loaded pipeline status |
| `/languages` | `GET` | Supported languages and index statistics |
| `/sample-queries` | `GET` | Sample queries for Gujarati and Hindi |
