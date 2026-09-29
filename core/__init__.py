"""
Core Package
============
Shared configuration and telemetry for the Voice RAG system.
"""

from core.config import (
    BASE_DIR,
    BUILDER_DIR,
    DATA_DIR,
    STATIC_DIR,
    DEVICE,
    USE_FP16,
    GROQ_MODEL_NAME,
    BGE_MODEL_NAME,
    RERANKER_MODEL_NAME,
    DEFAULT_JUDGE_MODEL,
    LANGUAGE_METADATA,
    SYSTEM_PROMPTS,
    FINAL_TOP_K,
    DENSE_TOP_N,
    SPARSE_TOP_N,
    RRF_TOP_N,
    RRF_K,
)
