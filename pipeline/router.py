"""
Multilingual Language Router
============================
Central coordinator for Multi-Language Voice & Text RAG.
Manages language pipelines, the shared BGE-M3 model and the shared reranker, and provides
smart script-based auto routing.
"""

import os
import gc
import warnings
import torch
import faiss
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Generator
from groq import Groq

from core.config import (
    BUILDER_DIR,
    DATA_DIR,
    DEVICE,
    USE_FP16,
    LANGUAGE_METADATA,
    FINAL_TOP_K,
    BGE_MODEL_NAME,
)
from pipeline.embeddings import NativeBGEM3
from pipeline.reranker import RAGReranker
from pipeline.single_pipeline import SingleLanguagePipeline
from pipeline.generator import (
    normalize_lang_code,
    detect_script_language,
    warmup_llm,
)

_NO_PIPELINES_MSG = (
    "No RAG language pipelines are loaded. Check RAG_LANGUAGES and that "
    "voice_rag_builder/<language>/ contains pipeline_manifest.json, the FAISS index, the sparse "
    "matrix and passage_metadata.sqlite (see the startup log for the load error)."
)


class LanguageRouter:
    """
    Central Language Router for Voice & Text RAG.
    Loads and manages language pipelines (Gujarati, Hindi, etc.) sharing a single BGE-M3 model
    and a single reranker. Routes queries, streaming, retrieval, and evaluation to the selected
    language.
    """

    def __init__(
        self,
        base_builder_dir: Optional[Path] = None,
        base_eval_dir: Optional[Path] = None,
        verbose: bool = True,
        reranker: Optional[RAGReranker] = None,
    ):
        self.verbose = verbose
        root_dir = Path(__file__).resolve().parent.parent

        self.builder_dir = Path(base_builder_dir) if base_builder_dir else BUILDER_DIR
        if base_eval_dir:
            self.eval_dir = Path(base_eval_dir)
        elif DATA_DIR.exists():
            self.eval_dir = DATA_DIR
        else:
            self.eval_dir = root_dir / "Eval"

        # 1. Connect Groq client
        api_key = (os.environ.get("GROQ_API_KEY") or "").strip().strip("\"'") or None
        self.groq_client = Groq(api_key=api_key) if api_key else None

        # 2. One reranker for every language (stateless; cross-encoder weights are a process singleton)
        self.reranker = reranker or RAGReranker()

        # 3. Load Shared BGE-M3 Query Encoder
        if self.verbose:
            print("=" * 60)
            print(f"🚀 [LanguageRouter] Initializing Shared Multilingual Encoder ({BGE_MODEL_NAME})...")
            print("=" * 60)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.bge_model = NativeBGEM3(
            model_name=BGE_MODEL_NAME,
            use_fp16=USE_FP16,
            devices=DEVICE,
        )

        # Warm-up pass
        _ = self.bge_model.encode(["નમસ્તે"], max_length=16)
        _ = self.bge_model.encode(["नमस्ते"], max_length=16)
        if self.verbose:
            q = " (int8 quantized)" if self.bge_model.quantized else ""
            print(f"✅ [LanguageRouter] Shared encoder loaded and warmed up{q}! torch threads={torch.get_num_threads()}")
            print(f"🔀 [LanguageRouter] Reranker backend: {self.reranker.backend}"
                  + (f" ({self.reranker.model_name})" if self.reranker.backend == "llm" else ""))

        # 4. Discover & Load Language Pipelines
        self.pipelines: Dict[str, SingleLanguagePipeline] = {}
        self._load_available_languages()
        if not self.pipelines:
            print(f"⚠️ [LanguageRouter] {_NO_PIPELINES_MSG}")

        # 5. Pre-open the Groq connection so the first query does not pay TLS handshake
        warmup_llm()

    def _load_available_languages(self):
        requested = os.environ.get("RAG_LANGUAGES", "gu,hi")
        candidates = [normalize_lang_code(c) for c in requested.split(",") if c.strip()]
        for code in dict.fromkeys(candidates):
            meta = LANGUAGE_METADATA.get(code, LANGUAGE_METADATA["gu"])
            folder_name = meta["folder"]
            lang_builder_dir = self.builder_dir / folder_name
            lang_eval_dir = self.eval_dir / folder_name

            # Fallback check
            if not lang_builder_dir.exists():
                lang_builder_dir = self.builder_dir

            if (lang_builder_dir / "pipeline_manifest.json").exists():
                try:
                    pipeline = SingleLanguagePipeline(
                        lang_code=code,
                        data_dir=lang_builder_dir,
                        eval_dir=lang_eval_dir if lang_eval_dir.exists() else None,
                        shared_bge_encoder=self.bge_model,
                        groq_client=self.groq_client,
                        verbose=self.verbose,
                        reranker=self.reranker,
                    )
                    self.pipelines[code] = pipeline
                    if self.verbose:
                        print(f"✅ [LanguageRouter] Registered language: {meta['name']} ({code.upper()})")
                except Exception as e:
                    print(f"⚠️ [LanguageRouter] Failed loading language {code}: {e}")

        # Optimize FAISS search threads
        faiss.omp_set_num_threads(1)

    def _require_pipelines(self) -> None:
        if not self.pipelines:
            raise RuntimeError(_NO_PIPELINES_MSG)

    def resolve_language(
        self,
        query: Optional[str] = None,
        requested_lang: Optional[str] = "gu",
        auto_detect: bool = True
    ) -> str:
        """
        Smart language resolution:
        1. If requested_lang is 'auto' or auto_detect is True, check query script (Gujarati vs Hindi).
        2. If detected script matches a registered pipeline, use it.
        3. Otherwise normalize requested_lang or fall back to default ('gu').
        Raises RuntimeError when no pipeline is loaded.
        """
        self._require_pipelines()
        req_clean = (requested_lang or "").strip().lower()
        if (req_clean == "auto" or auto_detect) and query:
            detected = detect_script_language(query)
            if detected and detected in self.pipelines:
                return detected

        if req_clean and req_clean != "auto":
            normalized = normalize_lang_code(req_clean)
            if normalized in self.pipelines:
                return normalized

        # Fallback to default registered pipeline
        return "gu" if "gu" in self.pipelines else next(iter(self.pipelines))

    def get_pipeline(self, lang: Optional[str] = "gu") -> SingleLanguagePipeline:
        self._require_pipelines()
        code = normalize_lang_code(lang)
        if code in self.pipelines:
            return self.pipelines[code]
        return next(iter(self.pipelines.values()))

    def get_available_languages(self) -> List[dict]:
        res = []
        for code, meta in LANGUAGE_METADATA.items():
            pipeline = self.pipelines.get(code)
            if pipeline:
                golden_count = len(pipeline.evaluator.golden_records) if pipeline.evaluator else 0
                res.append({
                    "code": code,
                    "name": meta["name"],
                    "native_name": meta["native_name"],
                    "locale": meta["locale"],
                    "indexed_passages": int(pipeline.dense_index.ntotal),
                    "golden_dataset_records": golden_count,
                    "is_active": True,
                })
        return res

    def ask(
        self,
        query: str,
        lang: Optional[str] = "gu",
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
        history: Optional[Iterable[Any]] = None,
    ) -> dict:
        actual_lang = self.resolve_language(query=query, requested_lang=lang)
        pipeline = self.get_pipeline(actual_lang)
        return pipeline.ask(query=query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by, history=history)

    def ask_stream(
        self,
        query: str,
        lang: Optional[str] = "gu",
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
        history: Optional[Iterable[Any]] = None,
    ) -> Generator[dict, None, None]:
        actual_lang = self.resolve_language(query=query, requested_lang=lang)
        pipeline = self.get_pipeline(actual_lang)
        yield from pipeline.ask_stream(query=query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by, history=history)

    def hybrid_retrieve(
        self,
        query: str,
        lang: Optional[str] = "gu",
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
    ) -> dict:
        actual_lang = self.resolve_language(query=query, requested_lang=lang)
        pipeline = self.get_pipeline(actual_lang)
        return pipeline.hybrid_retrieve(query=query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by)

    def evaluate(self, *args, lang: Optional[str] = "gu", **kwargs) -> dict:
        pipeline = self.get_pipeline(lang)
        return pipeline.evaluate(*args, **kwargs)

    def ask_and_eval(
        self,
        query: str,
        lang: Optional[str] = "gu",
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
        history: Optional[Iterable[Any]] = None,
        query_id: Optional[int] = None,
    ) -> dict:
        actual_lang = self.resolve_language(query=query, requested_lang=lang)
        pipeline = self.get_pipeline(actual_lang)
        rag_res = pipeline.ask(query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by, history=history)
        eval_res = pipeline.evaluate(
            query=query,
            answer=rag_res["answer"],
            documents=rag_res["documents"],
            query_id=query_id,
        )
        rag_res["evaluation"] = eval_res
        return rag_res

    def get_sample_queries(self, lang: Optional[str] = "gu", count: int = 20) -> list:
        pipeline = self.get_pipeline(lang)
        return pipeline.get_sample_queries(count=count)

    def close(self):
        for p in self.pipelines.values():
            p.close()


class GujaratiHybridRAGNoReranker:
    """
    DEPRECATED: backward-compatible wrapper kept for legacy scripts (chatbot.py,
    measure_latency.py). Use `LanguageRouter` (or `LanguageRouter().get_pipeline("gu")`).
    Routes directly to the LanguageRouter's Gujarati pipeline with the legacy method signatures.
    """

    def __init__(self, data_dir=None, golden_dataset_path=None, load_groq=True, verbose=True):
        warnings.warn(
            "GujaratiHybridRAGNoReranker is deprecated; use pipeline.router.LanguageRouter.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._router = LanguageRouter(verbose=verbose)
        self._pipe = self._router.get_pipeline("gu")

        self.dense_index = self._pipe.dense_index
        self.meta_conn = self._pipe.meta_conn
        self.bge_model = self._pipe.bge_model
        self.bge_model_name = BGE_MODEL_NAME
        self.groq_client = self._router.groq_client
        self.evaluator = self._pipe.evaluator

    def ask(self, query, final_k=FINAL_TOP_K, use_reranker=False, sort_by="rrf", history=None):
        return self._pipe.ask(query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by, history=history)

    def ask_stream(self, query, final_k=FINAL_TOP_K, use_reranker=False, sort_by="rrf", history=None):
        yield from self._pipe.ask_stream(query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by, history=history)

    def hybrid_retrieve(self, query, final_k=FINAL_TOP_K, use_reranker=False, sort_by="rrf"):
        return self._pipe.hybrid_retrieve(query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by)

    def evaluate(self, *args, **kwargs):
        return self._pipe.evaluate(*args, **kwargs)

    def ask_and_eval(self, query, final_k=FINAL_TOP_K):
        return self._router.ask_and_eval(query, lang="gu", final_k=final_k)

    def get_sample_queries(self, count=20):
        return self._pipe.get_sample_queries(count=count)

    def close(self):
        self._router.close()
