# ==============================================================================
# Multi-Language Voice & Text RAG System (v3) - Production Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Prevent interactive prompts & buffer stdout/stderr
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install essential system dependencies (C++ compilers for FAISS, audio tools, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install PyTorch CPU and Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code & static assets
COPY . .

# Expose standard FastAPI application port
EXPOSE 8000

# Health check to ensure pipeline loaded in lifespan
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
