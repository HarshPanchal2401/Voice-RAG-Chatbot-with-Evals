"""
Pipeline Package
================
Modular RAG pipeline components:
- embeddings: Native BGE-M3 multilingual encoder (thread-safe LRU query cache)
- retriever: Hybrid FAISS HNSW + SciPy CSC + RRF fusion & sorting
- reranker: RAGReranker with llm / cross_encoder / lexical backends and score thresholds
- metadata_store: read-only SQLite passage/chunk metadata store (doc_key per chunk)
- generator: System prompts, context builder, conversation memory, query condensing,
  answer-language detection, sentence splitting, Groq streaming
- single_pipeline: SingleLanguagePipeline coordinating one language
- router: Multilingual LanguageRouter with auto script detection
"""
import core.config
from pipeline.embeddings import NativeBGEM3
from pipeline.metadata_store import MetadataStore, make_doc_key
from pipeline.reranker import RAGReranker, lexical_tokens, lexical_score
from pipeline.retriever import HybridRetriever
from pipeline.generator import (
    normalize_lang_code,
    normalize_answer_lang,
    detect_script_language,
    detect_answer_language,
    split_sentences,
    strip_citations,
    is_no_answer,
    no_answer_message,
    normalize_history,
    build_context,
    build_user_prompt,
    build_messages,
    condense_query,
    stream_answer,
    generate_answer,
    warmup_llm,
)
from pipeline.single_pipeline import SingleLanguagePipeline
from pipeline.router import LanguageRouter, GujaratiHybridRAGNoReranker

__all__ = [
    "NativeBGEM3",
    "MetadataStore",
    "make_doc_key",
    "RAGReranker",
    "lexical_tokens",
    "lexical_score",
    "HybridRetriever",
    "normalize_lang_code",
    "normalize_answer_lang",
    "detect_script_language",
    "detect_answer_language",
    "split_sentences",
    "strip_citations",
    "is_no_answer",
    "no_answer_message",
    "normalize_history",
    "build_context",
    "build_user_prompt",
    "build_messages",
    "condense_query",
    "stream_answer",
    "generate_answer",
    "warmup_llm",
    "SingleLanguagePipeline",
    "LanguageRouter",
    "GujaratiHybridRAGNoReranker",
]
