"""
Single Language Pipeline Instance
=================================
Dedicated pipeline for a single language (Index + Sparse Matrix + SQLite + Evaluator + Reranker).
Uses the shared multilingual BGE-M3 Query Encoder and (optionally) a shared reranker.
"""

import gc
import json
import unicodedata
import re
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Generator

import faiss
from scipy import sparse
from groq import Groq

from core.config import (
    LANGUAGE_METADATA,
    SYSTEM_PROMPTS,
    DEFAULT_HNSW_EF_SEARCH,
    BGE_QUERY_MAX_LENGTH,
    FINAL_TOP_K,
    CONDENSE_QUERY,
)
from core.tracing import traceable, add_run_metadata, current_trace_id
from pipeline.embeddings import NativeBGEM3
from pipeline.metadata_store import MetadataStore, make_doc_key
from pipeline.reranker import RAGReranker
from pipeline.retriever import HybridRetriever
from pipeline.generator import (
    build_context as _build_context,
    build_user_prompt as _build_user_prompt,
    condense_query,
    detect_answer_language,
    normalize_history,
    stream_answer,
)
from services.evaluation_service import DeepEvalEvaluator


def _stream_reduce(items: List[dict]) -> dict:
    for item in reversed(items):
        if item.get("type") == "meta":
            return {k: v for k, v in item.items() if k != "documents"}
    return {}


class SingleLanguagePipeline:
    """
    Dedicated pipeline instance for a single language.
    Orchestrates query condensing, retrieval, sorting, re-ranking, prompt assembly,
    generation, and evaluation.
    """

    def __init__(
        self,
        lang_code: str,
        data_dir: Path,
        eval_dir: Optional[Path],
        shared_bge_encoder: NativeBGEM3,
        groq_client: Optional[Groq] = None,
        verbose: bool = True,
        reranker: Optional[RAGReranker] = None,
    ):
        self.lang_code = lang_code
        self.meta = LANGUAGE_METADATA.get(lang_code, LANGUAGE_METADATA["gu"])
        self.data_dir = Path(data_dir)
        self.eval_dir = Path(eval_dir) if eval_dir else None
        self.bge_model = shared_bge_encoder
        self.groq_client = groq_client
        self.verbose = verbose
        self.system_prompt = SYSTEM_PROMPTS.get(lang_code, SYSTEM_PROMPTS["gu"])
        self._reranker = reranker

        self._resolve_paths()
        self._load_manifest()
        self._load_stores()
        self._load_evaluator()

        # Initialize Retriever
        self.retriever = HybridRetriever(
            dense_index=self.dense_index,
            sparse_csc=self.doc_sparse_csc,
            sparse_dim=self.sparse_dim,
            metadata_store=self.metadata_store,
            lang_code=self.lang_code,
            reranker=self.reranker,
        )

    def _resolve_paths(self):
        d = self.data_dir
        self.manifest_path = d / "pipeline_manifest.json"
        compact = d / "bge_m3_dense_hnsw_fp16.index"
        self.dense_index_path = compact if compact.exists() else d / "bge_m3_dense_hnsw.index"
        self.sparse_matrix_path = d / "bge_m3_sparse_lexical.npz"
        self.meta_db_path = d / "passage_metadata.sqlite"

        required = [
            self.manifest_path,
            self.dense_index_path,
            self.sparse_matrix_path,
            self.meta_db_path,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Missing required artifact in {d}: {path}")

    def _load_manifest(self):
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.dense_dim = int(self.manifest.get("dense_dim", 1024))
        self.sparse_dim = int(self.manifest.get("sparse_dim", 250002))
        dense_cfg = self.manifest.get("dense_index", {})
        self.hnsw_ef_search = int(dense_cfg.get("default_efSearch", DEFAULT_HNSW_EF_SEARCH))

    def _load_stores(self):
        if self.verbose:
            print(f"⏳ [{self.meta['name']}] Loading FAISS dense index ({self.dense_index_path.name})...")
        self.dense_index = faiss.read_index(str(self.dense_index_path))
        if hasattr(self.dense_index, "hnsw"):
            self.dense_index.hnsw.efSearch = self.hnsw_ef_search

        if self.verbose:
            print(f"⏳ [{self.meta['name']}] Loading sparse lexical matrix...")
        self.doc_sparse_csc = sparse.load_npz(self.sparse_matrix_path).tocsc()
        self.doc_sparse_csc.sort_indices()
        self.num_docs = self.doc_sparse_csc.shape[0]
        gc.collect()

        self.metadata_store = MetadataStore(self.meta_db_path, lang_code=self.lang_code)
        # Keep legacy conn reference for backward compatibility
        self.meta_conn = self.metadata_store.conn

        metadata_count = self.metadata_store.count_chunks()
        if self.verbose:
            print(f"📊 [{self.meta['name']}] Indexed Passages: {self.dense_index.ntotal:,} vectors ({metadata_count:,} chunks).")

    def _load_evaluator(self):
        eval_path = None
        if self.eval_dir and self.eval_dir.exists():
            for candidate in ("golden_dataset_full.jsonl", "golden_dataset_sample_500.jsonl", "golden_dataset_sample_100.jsonl"):
                cand_path = self.eval_dir / candidate
                if cand_path.exists():
                    eval_path = str(cand_path)
                    break

        self.evaluator = DeepEvalEvaluator(
            golden_dataset_path=eval_path,
            groq_client=self.groq_client,
            embedding_model=self.bge_model,
            language=self.lang_code
        ) if eval_path else None

    @property
    def reranker(self) -> RAGReranker:
        if self._reranker is None:
            self._reranker = RAGReranker()
        return self._reranker

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", str(text))
        return re.sub(r"\s+", " ", text).strip()

    def fetch_metadata(self, vector_ids: List[int]) -> List[dict]:
        return self.metadata_store.fetch_by_vector_ids(vector_ids)

    @traceable(run_type="embedding", name="encode_query", process_outputs=lambda o: {"dense_dim": int(o[0].shape[1]), "sparse_terms": int(len(o[1][0]))})
    def encode_query(self, query: str):
        query = self.clean_text(query)
        dense, tok_ids, tok_w = self.bge_model.encode_hybrid(query, max_length=BGE_QUERY_MAX_LENGTH)
        faiss.normalize_L2(dense)  # encoder returns a private copy, so in-place is safe
        valid = (tok_ids >= 0) & (tok_ids < self.sparse_dim)
        return dense, (tok_ids[valid], tok_w[valid])

    def dense_search(self, query_dense, top_n: int = 50) -> List[dict]:
        return self.retriever.dense_search(query_dense, top_n=top_n)

    def sparse_search(self, query_sparse, top_n: int = 50) -> List[dict]:
        return self.retriever.sparse_search(query_sparse, top_n=top_n)

    @staticmethod
    def reciprocal_rank_fusion(dense_results, sparse_results, rrf_k: int = 60, top_n: int = 30):
        return HybridRetriever.reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=rrf_k, top_n=top_n)

    def hybrid_retrieve(
        self,
        query: str,
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
    ) -> Dict[str, Any]:
        """
        Retrieves and sorts top passages for a query.
        Supports sorting by 'rrf', 'dense', 'sparse', or 'rerank'.
        """
        t0 = time.perf_counter()
        query_dense, query_sparse = self.encode_query(query)
        encode_ms = (time.perf_counter() - t0) * 1000

        retrieval = self.retriever.retrieve(
            query=query,
            query_dense=query_dense,
            query_sparse=query_sparse,
            final_k=final_k,
            use_reranker=use_reranker,
            sort_by=sort_by,
        )
        retrieval["timings"]["query_encoding_ms"] = encode_ms
        retrieval["timings"]["retrieval_total_ms"] += encode_ms
        return retrieval

    def build_context(self, documents: List[dict], answer_lang: Optional[str] = None) -> str:
        return _build_context([doc["text"] for doc in documents], answer_lang or self.lang_code)

    def build_user_prompt(self, query: str, context: str, answer_lang: Optional[str] = None) -> str:
        return _build_user_prompt(query, context, answer_lang or self.lang_code)

    @traceable(run_type="chain", name="rag_ask")
    def ask(
        self,
        query: str,
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
        history: Optional[Iterable[Any]] = None,
    ) -> dict:
        result: Dict[str, Any] = {}
        for item in self.ask_stream(query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by, history=history):
            if item["type"] == "meta":
                result = item
        result.pop("type", None)
        result["query"] = query
        return result

    @traceable(run_type="chain", name="rag_ask_stream", reduce_fn=_stream_reduce)
    def ask_stream(
        self,
        query: str,
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
        history: Optional[Iterable[Any]] = None,
    ) -> Generator[dict, None, None]:
        """
        Streams {"type": "token", "content"} items, then one {"type": "meta", ...} item with
        language, answer_language, retrieval_query, no_answer, answer, documents,
        retrieval_timings (incl. condense_ms), ttft_ms, llm_ms, total_ms, model, trace_id.
        """
        t_start = time.perf_counter()
        history = normalize_history(history)
        answer_language = detect_answer_language(query, self.lang_code)

        # 1. Follow-up -> standalone retrieval query (only with history)
        retrieval_query = query
        condense_ms: Optional[float] = None
        if history and CONDENSE_QUERY:
            t0 = time.perf_counter()
            retrieval_query = condense_query(query, history, lang=answer_language) or query
            condense_ms = (time.perf_counter() - t0) * 1000

        # 2. Retrieval (+ optional rerank; may return no documents -> no-answer path)
        retrieval = self.hybrid_retrieve(retrieval_query, final_k=final_k, use_reranker=use_reranker, sort_by=sort_by)
        retrieval["timings"]["condense_ms"] = condense_ms
        documents = retrieval["documents"]
        contexts = [d["text"] for d in documents if d.get("text")]

        # 3. Generation (the LLM sees the user's own question plus the conversation)
        done: Dict[str, Any] = {}
        for item in stream_answer(query, contexts, lang=self.lang_code, history=history, answer_lang=answer_language):
            if item["type"] == "token":
                yield item
            else:
                done = item

        total_ms = (time.perf_counter() - t_start) * 1000
        no_answer = bool(done.get("no_answer", not contexts))
        add_run_metadata(
            language=self.lang_code,
            answer_language=answer_language,
            model=done.get("model"),
            no_answer=no_answer,
            condensed=retrieval_query != query,
            ttft_ms=round(done.get("ttft_ms", 0.0), 2),
            llm_ms=round(done.get("llm_ms", 0.0), 2),
            retrieval_ms=round(retrieval["timings"]["retrieval_total_ms"], 2),
            condense_ms=round(condense_ms, 2) if condense_ms is not None else None,
            total_ms=round(total_ms, 2),
        )

        yield {
            "type": "meta",
            "language": self.lang_code,
            "answer_language": answer_language,
            "retrieval_query": retrieval_query,
            "no_answer": no_answer,
            "answer": done.get("answer", ""),
            "documents": documents,
            "retrieval_timings": retrieval["timings"],
            "ttft_ms": done.get("ttft_ms", 0.0),
            "llm_ms": done.get("llm_ms", 0.0),
            "total_ms": total_ms,
            "model": done.get("model"),
            "trace_id": current_trace_id(),
        }

    @staticmethod
    def _doc_key(d: Dict[str, Any]) -> Optional[str]:
        if d.get("doc_key"):
            return d["doc_key"]
        if d.get("query_id") is not None and d.get("passage_id") is not None:
            return make_doc_key(d["query_id"], d["passage_id"])
        return None

    @traceable(run_type="chain", name="deepeval_evaluate")
    def evaluate(
        self,
        query: str = "",
        answer: str = "",
        documents: Optional[List[Dict[str, Any]]] = None,
        query_id: Optional[int] = None,
        allow_open_eval: bool = True,
    ) -> dict:
        if not self.evaluator:
            return {
                "is_golden": False,
                "query_id": query_id,
                "query_type": None,
                "ground_truth_answer": None,
                "warning": "⚠️ Evaluator not initialized or golden dataset not found.",
                "scores": None,
                "failed_metrics": [],
            }

        documents = documents or []
        contexts = [d.get("text", "") for d in documents]
        passage_ids = [d["passage_id"] if d.get("passage_id") is not None else d.get("chunk_id") for d in documents]
        doc_keys = [self._doc_key(d) for d in documents]

        kwargs = dict(
            question=query,
            generated_answer=answer,
            retrieved_contexts=contexts,
            retrieved_passage_ids=passage_ids,
            query_id=query_id,
            allow_open_eval=allow_open_eval,
        )
        try:
            return self.evaluator.evaluate(**kwargs, retrieved_doc_keys=doc_keys)
        except TypeError as e:
            if "retrieved_doc_keys" not in str(e):
                raise
            # Evaluator without doc-key support (older services/evaluation_service.py)
            return self.evaluator.evaluate(**kwargs)

    def get_sample_queries(self, count: int = 20) -> list:
        if self.evaluator:
            return self.evaluator.get_random_samples(count=count)
        return []

    def close(self):
        if hasattr(self, "metadata_store") and self.metadata_store:
            self.metadata_store.close()
