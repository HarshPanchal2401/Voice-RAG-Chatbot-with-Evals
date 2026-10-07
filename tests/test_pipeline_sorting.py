"""
Unit & Integration Tests for Pipeline Sorting & Ranking
======================================================
Validates that document rankings, RRF fusion, multi-criteria sorting,
and re-ranker stages correctly order candidates.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.retriever import HybridRetriever
from pipeline.generator import detect_script_language


def test_rrf_scoring_order():
    dense_results = [
        {"vector_id": 101, "dense_rank": 1, "dense_score": 0.95},
        {"vector_id": 102, "dense_rank": 2, "dense_score": 0.85},
        {"vector_id": 103, "dense_rank": 3, "dense_score": 0.75},
    ]
    sparse_results = [
        {"vector_id": 102, "sparse_rank": 1, "sparse_score": 12.5},
        {"vector_id": 104, "sparse_rank": 2, "sparse_score": 10.0},
        {"vector_id": 101, "sparse_rank": 3, "sparse_score": 8.0},
    ]

    fused = HybridRetriever.reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60, top_n=10)

    # Validate output length
    assert len(fused) == 4

    # Validate strictly sorted descending by rrf_score
    scores = [x["rrf_score"] for x in fused]
    assert scores == sorted(scores, reverse=True)

    # Document 102 was rank 2 dense & rank 1 sparse:
    # 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
    # Document 101 was rank 1 dense & rank 3 sparse:
    # 1/(60+1) + 1/(60+3) = 0.016393 + 0.015873 = 0.032266
    # 102 must rank above 101
    assert fused[0]["vector_id"] == 102
    assert fused[1]["vector_id"] == 101


def test_multi_mode_document_sorting():
    docs = [
        {"vector_id": 1, "rrf_score": 0.03, "dense_score": 0.70, "sparse_score": 15.0, "rerank_score": 0.2},
        {"vector_id": 2, "rrf_score": 0.05, "dense_score": 0.90, "sparse_score": 5.0, "rerank_score": 0.5},
        {"vector_id": 3, "rrf_score": 0.01, "dense_score": 0.80, "sparse_score": 25.0, "rerank_score": 1.0},
    ]

    # Test sorting by RRF
    by_rrf = HybridRetriever.sort_documents(docs, sort_by="rrf")
    assert [d["vector_id"] for d in by_rrf] == [2, 1, 3]
    assert [d["rank"] for d in by_rrf] == [1, 2, 3]

    # Test sorting by Dense
    by_dense = HybridRetriever.sort_documents(docs, sort_by="dense")
    assert [d["vector_id"] for d in by_dense] == [2, 3, 1]
    assert [d["rank"] for d in by_dense] == [1, 2, 3]

    # Test sorting by Sparse
    by_sparse = HybridRetriever.sort_documents(docs, sort_by="sparse")
    assert [d["vector_id"] for d in by_sparse] == [3, 1, 2]
    assert [d["rank"] for d in by_sparse] == [1, 2, 3]

    # Test sorting by Rerank
    by_rerank = HybridRetriever.sort_documents(docs, sort_by="rerank")
    assert [d["vector_id"] for d in by_rerank] == [3, 2, 1]
    assert [d["rank"] for d in by_rerank] == [1, 2, 3]


def test_script_detection():
    assert detect_script_language("વાઇનની બોટલમાં કેટલા ઔંસ હોય છે") == "gu"
    assert detect_script_language("क्या वह शादीशुदा है?") == "hi"
    assert detect_script_language("Hello world") is None


if __name__ == "__main__":
    test_rrf_scoring_order()
    test_multi_mode_document_sorting()
    test_script_detection()
    print("✅ All sorting and script detection unit tests passed!")
