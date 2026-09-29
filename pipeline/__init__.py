"""
Pipeline Package
================
Modular RAG pipeline components:
- embeddings: Native BGE-M3 multilingual encoder
- retriever: Hybrid FAISS HNSW + SciPy CSC + RRF fusion & sorting
- reranker: Multilingual LLM & lexical listwise re-ranker
- metadata_store: SQLite passage/chunk metadata store
- generator: System prompts, context builder, Groq streaming
- single_pipeline: SingleLanguagePipeline coordinating one language
- router: Multilingual LanguageRouter with auto script detection
"""
import core.config
from pipeline.embeddings import NativeBGEM3
from pipeline.metadata_store import MetadataStore
from pipeline.reranker import RAGReranker
from pipeline.retriever import HybridRetriever
from pipeline.generator import (
    normalize_lang_code,
    detect_script_language,
    build_context,
    build_user_prompt,
    stream_answer,
    generate_answer,
    warmup_llm,
)
from pipeline.single_pipeline import SingleLanguagePipeline
from pipeline.router import LanguageRouter, GujaratiHybridRAGNoReranker

__all__ = [
    "NativeBGEM3",
    "MetadataStore",
    "RAGReranker",
    "HybridRetriever",
    "normalize_lang_code",
    "detect_script_language",
    "build_context",
    "build_user_prompt",
    "stream_answer",
    "generate_answer",
    "warmup_llm",
    "SingleLanguagePipeline",
    "LanguageRouter",
    "GujaratiHybridRAGNoReranker",
]
