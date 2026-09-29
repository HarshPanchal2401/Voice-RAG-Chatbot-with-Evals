"""
Hybrid Retriever & Ranking Engine
=================================
Performs dense FAISS HNSW search, sparse SciPy CSC inverted-index search,
reciprocal rank fusion (RRF), and deterministic multi-criteria sorting.
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
)
import faiss
from scipy import sparse
from core.tracing import traceable, as_langsmith_documents, add_run_metadata
from pipeline.metadata_store import MetadataStore
from pipeline.reranker import RAGReranker


def _retriever_outputs(outputs: dict) -> dict:
    if isinstance(outputs, dict) and "documents" in outputs:
        return {
            "documents": as_langsmith_documents(outputs["documents"]),
            "timings": outputs.get("timings"),
            "language": outputs.get("language"),
        }
    return outputs


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
    ):
        self.dense_index = dense_index
        self.sparse_csc = sparse_csc
        self.sparse_dim = sparse_dim
        self.metadata_store = metadata_store
        self.lang_code = lang_code
        self.reranker = reranker or RAGReranker()

    @traceable(run_type="tool", name="dense_search", process_inputs=lambda i: {"top_n": i.get("top_n")})
    def dense_search(self, query_dense: np.ndarray, top_n: int = DENSE_TOP_N) -> List[Dict[str, Any]]:
        """Dense search using FAISS HNSW cosine/L2 normalized index."""
        scores, ids = self.dense_index.search(query_dense, top_n)
        results = []
        for rank, (doc_id, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if doc_id >= 0:
                results.append({
                    "vector_id": int(doc_id),
                    "dense_score": float(score),
                    "dense_rank": rank,
                })
        return results

    @traceable(run_type="tool", name="sparse_search", process_inputs=lambda i: {"top_n": i.get("top_n")})
    def sparse_search(self, query_sparse: Tuple[np.ndarray, np.ndarray], top_n: int = SPARSE_TOP_N) -> List[Dict[str, Any]]:
        """Sparse lexical search using CSC inverted-index columns."""
        tok_ids, tok_w = query_sparse
        if len(tok_ids) == 0:
            return []

        sub = self.sparse_csc[:, tok_ids]
        scores = np.asarray(sub @ tok_w.astype(np.float32)).ravel()
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
        Strictly sorts descending by RRF score.
        """
        merged: Dict[int, Dict[str, Any]] = {}
        for item in dense_results:
            vid = item["vector_id"]
            row = merged.setdefault(vid, {
                "vector_id": vid,
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
            })
            row["dense_rank"] = item["dense_rank"]
            row["dense_score"] = item["dense_score"]
            row["rrf_score"] += 1.0 / (rrf_k + item["dense_rank"])

        for item in sparse_results:
            vid = item["vector_id"]
            row = merged.setdefault(vid, {
                "vector_id": vid,
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
            })
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
          - 'rerank': Sort by rerank_score descending.
        """
        key = sort_by.lower().strip()
        if key == "dense":
            docs = sorted(documents, key=lambda d: (d.get("dense_score") is not None, d.get("dense_score") or -1.0), reverse=True)
        elif key == "sparse":
            docs = sorted(documents, key=lambda d: (d.get("sparse_score") is not None, d.get("sparse_score") or -1.0), reverse=True)
        elif key == "rerank":
            docs = sorted(documents, key=lambda d: (d.get("rerank_score") is not None, d.get("rerank_score") or -1.0), reverse=True)
        else:  # default 'rrf'
            docs = sorted(documents, key=lambda d: d.get("rrf_score", 0.0), reverse=True)

        # Re-assign sequential rank (1..N) according to the active sort order
        for idx, doc in enumerate(docs, start=1):
            doc["rank"] = idx
        return docs

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
        4. Metadata lookup from SQLite
        5. Optional listwise re-ranking via LLM / cross-scorer
        6. Deterministic sorting and top-k slicing
        """
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()

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
        fused = self.reciprocal_rank_fusion(dense_results, sparse_results, top_n=RRF_TOP_N)
        timings["rrf_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Metadata Lookup
        t0 = time.perf_counter()
        fused_ids = [x["vector_id"] for x in fused]
        docs = self.metadata_store.fetch_by_vector_ids(fused_ids)
        fused_map = {x["vector_id"]: x for x in fused}

        # Merge retrieval metrics with metadata
        for d in docs:
            info = fused_map.get(d["vector_id"], {})
            d.update({
                "rrf_score": info.get("rrf_score", 0.0),
                "dense_rank": info.get("dense_rank"),
                "sparse_rank": info.get("sparse_rank"),
                "dense_score": info.get("dense_score"),
                "sparse_score": info.get("sparse_score"),
            })
        timings["metadata_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Optional Re-ranking
        if use_reranker and self.reranker:
            candidate_pool = docs[:max(10, final_k * 2)]
            docs, rerank_ms = self.reranker.rerank(query, candidate_pool, top_k=final_k)
            timings["rerank_ms"] = rerank_ms
            # When reranking is active, documents are ordered by reranker
            docs = self.sort_documents(docs, sort_by="rerank")
        else:
            # Apply requested sorting strategy
            docs = self.sort_documents(docs, sort_by=sort_by)
            docs = docs[:final_k]

        timings["retrieval_total_ms"] = (time.perf_counter() - t_start) * 1000
        add_run_metadata(language=self.lang_code, **{k: round(v, 2) for k, v in timings.items()})

        return {
            "documents": docs,
            "timings": timings,
            "language": self.lang_code,
        }
