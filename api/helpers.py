"""
API Route Formatting Helpers
============================
Helper functions to convert pipeline outputs into typed Pydantic response models.
"""

import time
from typing import Optional, List, Dict, Any

from schemas.models import (
    SourceDocument,
    EvaluationScores,
    EvaluationResult,
    RetrievalTimings,
    LatencyBreakdown,
)
from core.tracing import log_eval_feedback
from pipeline.single_pipeline import SingleLanguagePipeline


def format_sources(raw_docs: list, lang: str = "gu") -> List[SourceDocument]:
    """Converts raw document dicts into validated SourceDocument instances."""
    formatted = []
    for rank, doc in enumerate(raw_docs, start=1):
        formatted.append(SourceDocument(
            rank=doc.get("rank", rank),
            chunk_id=doc.get("chunk_id", 0),
            passage_id=doc.get("passage_id", 0),
            url=doc.get("url"),
            title=doc.get("title"),
            section=doc.get("section"),
            text=doc.get("text", ""),
            language=doc.get("language", lang),
            rrf_score=round(doc.get("rrf_score", 0.0), 6),
            dense_rank=doc.get("dense_rank"),
            sparse_rank=doc.get("sparse_rank"),
            dense_score=round(float(doc["dense_score"]), 4) if doc.get("dense_score") is not None else None,
            sparse_score=round(float(doc["sparse_score"]), 4) if doc.get("sparse_score") is not None else None,
            rerank_score=doc.get("rerank_score"),
            rerank_rank=doc.get("rerank_rank"),
            original_rank=doc.get("original_rank"),
            rerank_reason=doc.get("rerank_reason"),
        ))
    return formatted


def format_evaluation(eval_raw: Optional[dict]) -> Optional[EvaluationResult]:
    """Converts evaluation dictionary into EvaluationResult."""
    if not eval_raw:
        return None

    scores = None
    if eval_raw.get("scores"):
        s = eval_raw["scores"]
        scores = EvaluationScores(
            overall_score=s.get("overall_score"),
            faithfulness=s.get("faithfulness"),
            answer_relevance=s.get("answer_relevance"),
            answer_correctness=s.get("answer_correctness"),
            answer_similarity=s.get("answer_similarity"),
            context_recall=s.get("context_recall"),
            hit_rate_at_k=s.get("hit_rate_at_k"),
            explanation=s.get("explanation"),
            reason=s.get("reason"),
        )

    return EvaluationResult(
        is_golden=eval_raw.get("is_golden", False),
        query_id=eval_raw.get("query_id"),
        query_type=eval_raw.get("query_type"),
        ground_truth_answer=eval_raw.get("ground_truth_answer"),
        warning=eval_raw.get("warning"),
        scores=scores,
    )


def build_latency(
    rag_result: dict,
    stt_ms: Optional[float] = None,
    eval_ms: Optional[float] = None,
    tts_ms: Optional[float] = None
) -> LatencyBreakdown:
    """Builds comprehensive LatencyBreakdown model."""
    retrieval_timings = rag_result.get("retrieval_timings") or rag_result.get("timings") or {}
    return LatencyBreakdown(
        stt_ms=stt_ms,
        retrieval=RetrievalTimings(**retrieval_timings),
        ttft_ms=rag_result.get("ttft_ms"),
        llm_ms=rag_result.get("llm_ms", 0.0),
        eval_ms=eval_ms,
        tts_ms=tts_ms,
        total_ms=(stt_ms or 0.0) + rag_result.get("total_ms", 0.0) + (tts_ms or 0.0),
    )



def run_evaluation(
    pipeline: SingleLanguagePipeline,
    query: str,
    rag_result: dict,
    query_id: Optional[int],
    trace_run_id: Optional[str]
) -> tuple:
    """DeepEval scoring + LangSmith feedback on the request trace. Returns (eval_result, eval_ms)."""
    t_eval = time.perf_counter()
    try:
        eval_result = pipeline.evaluate(
            query=query,
            answer=rag_result["answer"],
            documents=rag_result["documents"],
            query_id=query_id,
            allow_open_eval=True,
        )
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        return None, None
    eval_ms = (time.perf_counter() - t_eval) * 1000
    log_eval_feedback(trace_run_id, eval_result)
    return eval_result, eval_ms
