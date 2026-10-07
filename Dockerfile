# ==============================================================================
# Multi-Language Voice & Text RAG System (v3) - Production Dockerfile (CPU)
# ==============================================================================
# Build:  docker build -t voice-rag:v3 .
#         docker build --build-arg PRELOAD_BGE_M3=1 -t voice-rag:v3 .   # bake BGE-M3 into the image
# Run:    docker run -p 8000:8000 --env-file .env \
#           -v voice-rag-hf-cache:/opt/hf-cache \
#           voice-rag:v3
#
# Notes
# - The image needs the prebuilt indexes in `voice_rag_builder/<language>/`
#   (pipeline_manifest.json, bge_m3_dense_hnsw*.index, bge_m3_sparse_lexical.npz,
#   passage_metadata.sqlite). They are copied from the build context (~3.5 GB, see .dockerignore).
#   To keep the image small instead, exclude `voice_rag_builder/` in .dockerignore and mount it:
#   `-v "$PWD/voice_rag_builder:/app/voice_rag_builder:ro"`.
# - BGE-M3 (~2.3 GB) is downloaded to $HF_HOME on first start unless PRELOAD_BGE_M3=1.
#   Mount $HF_HOME as a volume (above) so restarts do not download it again.
# - Secrets are NEVER baked in: `.env` is excluded by .dockerignore; pass `--env-file .env`.
#   Set RAG_API_KEY when the port is reachable by anyone but you.
# - Needs ~6-8 GB RAM (BGE-M3 + two 1.3 GB HNSW indexes + SQLite page cache).
# - Behind a reverse proxy set FORWARDED_ALLOW_IPS=<proxy ip> so the per-IP rate limit sees real
#   client addresses (uvicorn trusts X-Forwarded-For only from those IPs).
# ==============================================================================
FROM python:3.11-slim

ARG PRELOAD_BGE_M3=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

# Runtime libs only: OpenMP for torch / faiss-cpu (both ship manylinux wheels, no compiler needed).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /app "$HF_HOME" \
    && chown -R app:app /app "$HF_HOME"

WORKDIR /app

# 1) CPU-only torch first (the default PyPI wheel pulls ~2 GB of CUDA libraries),
# 2) then the rest; `torch>=...` in requirements.txt is already satisfied by the CPU wheel.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Optional: bake BGE-M3 (tokenizer, weights, sparse head) into $HF_HOME.
RUN if [ "$PRELOAD_BGE_M3" = "1" ]; then \
      python -c "from transformers import AutoTokenizer, AutoModel; from huggingface_hub import hf_hub_download; \
AutoTokenizer.from_pretrained('BAAI/bge-m3'); AutoModel.from_pretrained('BAAI/bge-m3'); \
hf_hub_download('BAAI/bge-m3', 'sparse_linear.pt')" \
      && chown -R app:app "$HF_HOME"; \
    fi

# Application code, static UI, golden datasets and indexes (filtered by .dockerignore)
COPY --chown=app:app . .

USER app

EXPOSE 8000

# Model + ~2.6 GB of indexes load slowly: allow up to 10 minutes before health failures count.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).status == 200 else 1)"

# One worker: the models live in process memory (and the rate limiter is per process).
ENV PORT=8000
# Download indexes from RAG_INDEX_REPO when missing, then start the API (Spaces sets PORT)
CMD ["sh", "-c", "python scripts/fetch_indexes.py && exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
