"""
Hybrid Retriever & Ranking Engine
=================================
Performs dense FAISS HNSW search, sparse SciPy CSC inverted-index search,
reciprocal rank fusion (RRF), optional dense-score filtering, optional re-ranking
and deterministic multi-criteria sorting.
"""

import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from core.config import (
    DENSE_TOP_N,
    SPARSE_TOP_N,
    RRF_TOP_N,
    RRF_K,
    FINAL_TOP_K,
    RERANK_POOL,
    MIN_DENSE_SCORE,
)
import faiss
from scipy import sparse
from core.tracing import traceable, as_langsmith_documents, add_run_metadata
from pipeline.metadata_store import MetadataStore, make_doc_key
from pipeline.reranker import RAGReranker

# Fields every returned document carries (contract), with their defaults.
_DOC_DEFAULTS = {
    "rank": None,
    "rrf_score": 0.0,
    "dense_rank": None,
    "sparse_rank": None,
    "dense_score": None,
    "sparse_score": None,
    "rerank_score": None,
    "rerank_rank": None,
    "original_rank": None,
    "rerank_reason": None,
}


def _retriever_outputs(outputs: dict) -> dict:
    if isinstance(outputs, dict) and "documents" in outputs:
        return {
            "documents": as_langsmith_documents(outputs["documents"]),
            "timings": outputs.get("timings"),
            "language": outputs.get("language"),
        }
    return outputs


def _score_key(field: str):
    def key(d: Dict[str, Any]):
        value = d.get(field)
        return (value is not None, float(value) if value is not None else 0.0)
    return key


def dedupe_by_vector_id(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keeps the first occurrence of each vector_id (documents without one are kept)."""
    seen = set()
    out = []
    for d in documents:
        vid = d.get("vector_id")
        if vid is not None:
            if vid in seen:
                continue
            seen.add(vid)
        out.append(d)
    return out


class HybridRetriever:
    """
    Coordinates Dense, Sparse, RRF Fusion, Metadata, and Re-ranking with explicit sorting.
    """

    def __init__(
        self,
        dense_index: faiss.Index,
        sparse_csc: sparse.csc_matrix,
        sparse_dim: int,
        metadata_store: MetadataStore,
        lang_code: str = "gu",
        reranker: Optional[RAGReranker] = None,
        min_dense_score: Optional[float] = None,
        rerank_pool: Optional[int] = None,
    ):
        self.dense_index = dense_index
        self.sparse_csc = sparse_csc
        self.sparse_dim = sparse_dim
        self.metadata_store = metadata_store
        self.lang_code = lang_code
        self.reranker = reranker or RAGReranker()
        self.min_dense_score = MIN_DENSE_SCORE if min_dense_score is None else float(min_dense_score)
        self.rerank_pool = RERANK_POOL if rerank_pool is None else max(1, int(rerank_pool))

    @traceable(run_type="tool", name="dense_search", process_inputs=lambda i: {"top_n": i.get("top_n")})
    def dense_search(self, query_dense: np.ndarray, top_n: int = DENSE_TOP_N) -> List[Dict[str, Any]]:
        """Dense search using FAISS HNSW inner product on L2-normalized vectors (cosine)."""
        scores, ids = self.dense_index.search(query_dense, top_n)
        results = []
        seen = set()
        for doc_id, score in zip(ids[0], scores[0]):
            doc_id = int(doc_id)
            if doc_id < 0 or doc_id in seen:
                continue
            seen.add(doc_id)
            results.append({
                "vector_id": doc_id,
                "dense_score": float(score),
                "dense_rank": len(results) + 1,
            })
        return results

    @traceable(run_type="tool", name="sparse_search", process_inputs=lambda i: {"top_n": i.get("top_n")})
    def sparse_search(self, query_sparse: Tuple[np.ndarray, np.ndarray], top_n: int = SPARSE_TOP_N) -> List[Dict[str, Any]]:
        """Sparse lexical search using CSC inverted-index columns."""
        tok_ids, tok_w = query_sparse
        if len(tok_ids) == 0:
            return []

        sub = self.sparse_csc[:, tok_ids]
        scores = np.asarray(sub @ np.asarray(tok_w, dtype=np.float32)).ravel()
        cand = np.flatnonzero(scores)
        if cand.size == 0:
            return []

        values = scores[cand]
        k = min(top_n, values.size)
        top = np.argpartition(values, -k)[-k:] if k < values.size else np.arange(values.size)
        order = top[np.argsort(values[top])[::-1]]
        return [
            {
                "vector_id": int(cand[i]),
                "sparse_score": float(values[i]),
                "sparse_rank": rank,
            }
            for rank, i in enumerate(order, start=1)
        ]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        rrf_k: int = RRF_K,
        top_n: int = RRF_TOP_N,
    ) -> List[Dict[str, Any]]:
        """
        Combines Dense and Sparse rankings using Reciprocal Rank Fusion (RRF).
        Strictly sorts descending by RRF score (ties keep first-seen order). A vector_id that
        appears more than once in one list only counts its best (first) rank.
        """
        merged: Dict[int, Dict[str, Any]] = {}

        def _row(vid: int) -> Dict[str, Any]:
            return merged.setdefault(vid, {
                "vector_id": vid,
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
            })

        for item in dense_results:
            row = _row(item["vector_id"])
            if row["dense_rank"] is not None:
                continue
            row["dense_rank"] = item["dense_rank"]
            row["dense_score"] = item["dense_score"]
            row["rrf_score"] += 1.0 / (rrf_k + item["dense_rank"])

        for item in sparse_results:
            row = _row(item["vector_id"])
            if row["sparse_rank"] is not None:
                continue
            row["sparse_rank"] = item["sparse_rank"]
            row["sparse_score"] = item["sparse_score"]
            row["rrf_score"] += 1.0 / (rrf_k + item["sparse_rank"])

        # Strictly sort descending by rrf_score
        ranked = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:top_n]

    @staticmethod
    def sort_documents(documents: List[Dict[str, Any]], sort_by: str = "rrf") -> List[Dict[str, Any]]:
        """
        Enforces deterministic document sorting by requested criterion.
        Supported modes:
          - 'rrf': Sort by rrf_score descending.
          - 'dense': Sort by dense_score descending (None values pushed to bottom).
          - 'sparse': Sort by sparse_score descending (None values pushed to bottom).
          - 'rerank': Sort by rerank_score descending (None values pushed to bottom).
        Ties keep their input order. Ranks are re-assigned 1..N in the new order.
        """
        key = (sort_by or "rrf").lower().strip()
        if key in ("dense", "sparse", "rerank"):
            docs = sorted(documents, key=_score_key(f"{key}_score"), reverse=True)
        else:  # default 'rrf'
            docs = sorted(documents, key=lambda d: d.get("rrf_score") or 0.0, reverse=True)

        # Re-assign sequential rank (1..N) according to the active sort order
        for idx, doc in enumerate(docs, start=1):
            doc["rank"] = idx
        return docs

    def _fill_missing_dense_scores(self, query_dense: np.ndarray, docs: List[Dict[str, Any]]) -> None:
        """Exact cosine for sparse-only hits (vectors reconstructed from the index); best effort."""
        q = np.asarray(query_dense, dtype=np.float32).reshape(-1)
        for d in docs:
            if d.get("dense_score") is not None:
                continue
            try:
                vec = self.dense_index.reconstruct(int(d["vector_id"]))
                d["dense_score"] = float(np.dot(np.asarray(vec, dtype=np.float32).reshape(-1), q))
            except Exception:
                return  # index type without reconstruct support: leave unknown scores as None

    @traceable(run_type="retriever", name="hybrid_retrieve", process_outputs=_retriever_outputs)
    def retrieve(
        self,
        query: str,
        query_dense: np.ndarray,
        query_sparse: Tuple[np.ndarray, np.ndarray],
        final_k: int = FINAL_TOP_K,
        use_reranker: bool = False,
        sort_by: str = "rrf",
    ) -> Dict[str, Any]:
        """
        Complete Hybrid Retrieval Pipeline:
        1. Dense FAISS HNSW search
        2. Sparse SciPy CSC search
        3. Reciprocal Rank Fusion (RRF) & initial sort
        4. Metadata lookup from SQLite (+ doc_key), dedupe by vector_id
        5. Optional dense-score floor (RAG_MIN_DENSE_SCORE, off by default)
        6. Optional re-ranking of the top RAG_RERANK_POOL candidates (final order = rerank score;
           candidates under the reranker's minimum score are dropped, possibly all of them)
        7. Otherwise deterministic sorting and top-k slicing
        """
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        final_k = max(1, int(final_k))
        pool_size = max(self.rerank_pool, final_k)

        # 1. Dense Search
        t0 = time.perf_counter()
        dense_results = self.dense_search(query_dense, top_n=DENSE_TOP_N)
        timings["dense_search_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Sparse Search
        t0 = time.perf_counter()
        sparse_results = self.sparse_search(query_sparse, top_n=SPARSE_TOP_N)
        timings["sparse_search_ms"] = (time.perf_counter() - t0) * 1000

        # 3. RRF Fusion & Sorting
        t0 = time.perf_counter()
        fused = self.reciprocal_rank_fusion(
            dense_results, sparse_results,
            top_n=max(RRF_TOP_N, pool_size if use_reranker else final_k),
        )
        timings["rrf_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Metadata Lookup
        t0 = time.perf_counter()
        fused_map = {x["vector_id"]: x for x in fused}
        docs = dedupe_by_vector_id(self.metadata_store.fetch_by_vector_ids(list(fused_map)))

        # Merge retrieval metrics with metadata; original_rank = position in the RRF list
        rrf_pos = {vid: pos for pos, vid in enumerate(fused_map, start=1)}
        for d in docs:
            info = fused_map.get(d["vector_id"], {})
            for field, default in _DOC_DEFAULTS.items():
                d.setdefault(field, default)
            d.update({
                "rrf_score": info.get("rrf_score", 0.0),
                "dense_rank": info.get("dense_rank"),
                "sparse_rank": info.get("sparse_rank"),
                "dense_score": info.get("dense_score"),
                "sparse_score": info.get("sparse_score"),
                "original_rank": rrf_pos.get(d["vector_id"]),
            })
            if not d.get("doc_key"):
                d["doc_key"] = make_doc_key(d.get("query_id"), d.get("passage_id"))
        docs = self.sort_documents(docs, sort_by="rrf")
        timings["metadata_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Optional dense-score floor
        if self.min_dense_score > 0:
            self._fill_missing_dense_scores(query_dense, docs)
            before = len(docs)
            docs = [d for d in docs if d.get("dense_score") is None or d["dense_score"] >= self.min_dense_score]
            add_run_metadata(min_dense_score=self.min_dense_score, dense_filtered=before - len(docs))

        # 6./7. Re-ranking or sorting
        if use_reranker and self.reranker and docs:
            candidate_pool = docs[:pool_size]
            docs, rerank_ms = self.reranker.rerank(query, candidate_pool, top_k=final_k)
            timings["rerank_ms"] = rerank_ms
            # When reranking is active, documents are ordered by the reranker's score
            docs = self.sort_documents(docs, sort_by="rerank")[:final_k]
        else:
            # Apply requested sorting strategy
            docs = self.sort_documents(docs, sort_by=sort_by)[:final_k]

        timings["retrieval_total_ms"] = (time.perf_counter() - t_start) * 1000
        add_run_metadata(language=self.lang_code, **{k: round(v, 2) for k, v in timings.items()})

        return {
            "documents": docs,
            "timings": timings,
            "language": self.lang_code,
        }
