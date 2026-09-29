"""
Retrieval Component Metrics
===========================
Evaluates retrieval performance using:
1. Contextual Recall (DeepEval ContextualRecallMetric)
2. Contextual Precision (DeepEval ContextualPrecisionMetric)
3. Information Retrieval (IR) math metrics: Hit@K, Recall@K, MRR@K, NDCG@K, Precision@K
"""

import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric


# ============================================================
# Quantitative IR Math Metrics (Zero-LLM Fast Computations)
# ============================================================

def compute_hit_rate(retrieved_ids: List[int], gt_ids: List[int], k: int = 5) -> float:
    """Returns 1.0 if at least one ground-truth ID appears in top-K retrieved IDs, else 0.0."""
    if not gt_ids:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    hits = set(top_k_ids).intersection(set(gt_ids))
    return 1.0 if len(hits) > 0 else 0.0


def compute_recall_at_k(retrieved_ids: List[int], gt_ids: List[int], k: int = 5) -> float:
    """Returns the fraction of ground-truth IDs present in top-K retrieved IDs."""
    if not gt_ids:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    hits = set(top_k_ids).intersection(set(gt_ids))
    return round(len(hits) / len(set(gt_ids)), 4)


def compute_precision_at_k(retrieved_ids: List[int], gt_ids: List[int], k: int = 5) -> float:
    """Returns the fraction of top-K retrieved IDs that are ground-truth relevant."""
    if not retrieved_ids or k <= 0:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    hits = set(top_k_ids).intersection(set(gt_ids))
    return round(len(hits) / min(k, len(top_k_ids)), 4)


def compute_mrr(retrieved_ids: List[int], gt_ids: List[int], k: int = 10) -> float:
    """Returns Mean Reciprocal Rank (1/rank of first relevant item)."""
    gt_set = set(gt_ids)
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gt_set:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg(retrieved_ids: List[int], gt_ids: List[int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K with binary relevance."""
    gt_set = set(gt_ids)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gt_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all ground truth items ranked at top
    ideal_count = min(len(gt_set), k)
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def compute_context_jaccard(retrieved_contexts: List[str], gt_contexts: List[str]) -> float:
    """Word-level Jaccard overlap between retrieved passages and ground-truth text."""
    if not retrieved_contexts or not gt_contexts:
        return 0.0
    ret_words = set(re.findall(r"\w+", " ".join(retrieved_contexts).lower()))
    gt_words = set(re.findall(r"\w+", " ".join(gt_contexts).lower()))
    if not gt_words:
        return 0.0
    intersection = ret_words.intersection(gt_words)
    union = ret_words.union(gt_words)
    return round(len(intersection) / len(union), 4) if union else 0.0


# ============================================================
# DeepEval Contextual Recall & Precision (LLM-as-a-Judge)
# ============================================================

def _fallback_recall(
    retrieved_contexts: List[str],
    expected_output: str,
    gt_passage_ids: Optional[List[int]],
    retrieved_passage_ids: Optional[List[int]],
    gt_contexts: Optional[List[str]]
) -> Tuple[float, str]:
    has_pid_hit = False
    if gt_passage_ids and retrieved_passage_ids:
        has_pid_hit = len(set(retrieved_passage_ids).intersection(set(gt_passage_ids))) > 0

    all_ctx = " ".join(retrieved_contexts)
    gt_words = [w for w in re.findall(r"\w+", expected_output) if len(w) > 1]
    overlap = 0.0
    if gt_words:
        overlap = sum(1 for w in gt_words if w in all_ctx) / len(gt_words)

    if has_pid_hit and overlap >= 0.25:
        return 1.0, "Ground-truth passage retrieved and confirmed by factual overlap."
    elif has_pid_hit:
        return 0.95, "Ground-truth passage verified in retrieved candidate pool."
    elif overlap >= 0.5:
        return round(max(0.85, overlap), 4), "Heuristic overlap confirmed ground-truth facts in context."
    elif overlap > 0.0:
        return round(overlap, 4), "Partial factual overlap detected in retrieved context."
    return 0.0, "Context lacks ground-truth facts."


def _fallback_precision(
    retrieved_contexts: List[str],
    expected_output: str,
    gt_passage_ids: Optional[List[int]],
    retrieved_passage_ids: Optional[List[int]]
) -> Tuple[float, str]:
    if gt_passage_ids and retrieved_passage_ids:
        gt_set = set(gt_passage_ids)
        for rank, pid in enumerate(retrieved_passage_ids, start=1):
            if pid in gt_set:
                if rank == 1:
                    return 1.0, "Ground-truth passage ranked at position 1."
                elif rank == 2:
                    return 0.85, "Ground-truth passage ranked near top at position 2."
                elif rank == 3:
                    return 0.75, "Ground-truth passage ranked in top 3."
                elif rank == 4:
                    return 0.65, "Ground-truth passage ranked in top 4."
                else:
                    return round(max(0.55, 1.0 / rank), 4), f"Ground-truth passage retrieved at rank {rank}."

    # Lexical rank fallback
    gt_words = [w for w in re.findall(r"\w+", expected_output) if len(w) > 1]
    if gt_words and retrieved_contexts:
        scores = []
        for ctx in retrieved_contexts:
            ov = sum(1 for w in gt_words if w in ctx) / len(gt_words)
            scores.append(ov)
        if any(s > 0 for s in scores):
            best_rank = scores.index(max(scores)) + 1
            if best_rank == 1:
                return 0.95, "Most relevant lexical context ranked at position 1."
            elif best_rank == 2:
                return 0.80, "Most relevant lexical context ranked at position 2."
            elif best_rank == 3:
                return 0.70, "Most relevant lexical context ranked at position 3."
            elif best_rank == 4:
                return 0.60, "Most relevant lexical context ranked at position 4."
            else:
                return round(max(0.50, 1.0 / best_rank), 4), f"Most relevant context ranked at position {best_rank}."

    return 0.0, "No relevant retrieved context identified."


def compute_contextual_recall(
    question: str,
    retrieved_contexts: List[str],
    expected_output: str,
    model: Any,
    threshold: float = 0.7,
    gt_passage_ids: Optional[List[int]] = None,
    retrieved_passage_ids: Optional[List[int]] = None,
    gt_contexts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Measures Contextual Recall via DeepEval:
    Does the retrieved context contain all key facts required to produce expected output?
    """
    contexts = retrieved_contexts if retrieved_contexts else ["No context retrieved."]
    test_case = LLMTestCase(
        input=question,
        actual_output=expected_output,
        expected_output=expected_output,
        retrieval_context=contexts
    )

    metric = ContextualRecallMetric(threshold=threshold, model=model, include_reason=True)
    try:
        metric.measure(test_case)
        score = round(float(metric.score), 4)
        reason = metric.reason or "Context covers all essential facts."
        return {"score": score, "reason": reason, "success": score >= threshold}
    except Exception as e:
        fb_score, fb_reason = _fallback_recall(contexts, expected_output, gt_passage_ids, retrieved_passage_ids, gt_contexts)
        return {"score": fb_score, "reason": f"[heuristic fallback: judge failed] {fb_reason} (DeepEval notice: {str(e)})", "success": fb_score >= threshold, "fallback": True}


def compute_contextual_precision(
    question: str,
    retrieved_contexts: List[str],
    expected_output: str,
    model: Any,
    threshold: float = 0.7,
    gt_passage_ids: Optional[List[int]] = None,
    retrieved_passage_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Measures Contextual Precision via DeepEval:
    Are relevant context passages ranked above non-relevant passages?
    """
    contexts = retrieved_contexts if retrieved_contexts else ["No context retrieved."]
    test_case = LLMTestCase(
        input=question,
        actual_output=expected_output,
        expected_output=expected_output,
        retrieval_context=contexts
    )

    metric = ContextualPrecisionMetric(threshold=threshold, model=model, include_reason=True)
    try:
        metric.measure(test_case)
        score = round(float(metric.score), 4)
        reason = metric.reason or "Relevant passages ranked favorably."
        return {"score": score, "reason": reason, "success": score >= threshold}
    except Exception as e:
        fb_score, fb_reason = _fallback_precision(contexts, expected_output, gt_passage_ids, retrieved_passage_ids)
        return {"score": fb_score, "reason": f"[heuristic fallback: judge failed] {fb_reason} (DeepEval notice: {str(e)})", "success": fb_score >= threshold, "fallback": True}


def evaluate_retrieval_record(
    question: str,
    retrieved_contexts: List[str],
    retrieved_passage_ids: List[int],
    gt_contexts: List[str],
    gt_passage_ids: List[int],
    expected_output: str,
    model: Optional[Any] = None,
    k: int = 5,
    run_llm_metrics: bool = True
) -> Dict[str, Any]:
    """
    Evaluates retrieval strictly on Contextual Recall and Contextual Precision.
    """
    ctx_recall = {"score": 0.0, "reason": "Evaluated without LLM judge"}
    ctx_precision = {"score": 0.0, "reason": "Evaluated without LLM judge"}

    if run_llm_metrics and model:
        ctx_recall = compute_contextual_recall(
            question, retrieved_contexts, expected_output, model,
            gt_passage_ids=gt_passage_ids,
            retrieved_passage_ids=retrieved_passage_ids,
            gt_contexts=gt_contexts
        )
        ctx_precision = compute_contextual_precision(
            question, retrieved_contexts, expected_output, model,
            gt_passage_ids=gt_passage_ids,
            retrieved_passage_ids=retrieved_passage_ids
        )
    else:
        # Fallback to lexical/passage ID heuristics if LLM judge is not configured
        recall_k = compute_recall_at_k(retrieved_passage_ids, gt_passage_ids, k=k)
        precision_k = compute_precision_at_k(retrieved_passage_ids, gt_passage_ids, k=k)
        ctx_recall = {"score": recall_k, "reason": "Heuristic fallback via passage overlap"}
        ctx_precision = {"score": precision_k, "reason": "Heuristic fallback via precision @ K"}

    return {
        "contextual_recall": ctx_recall["score"],
        "contextual_recall_reason": ctx_recall.get("reason", ""),
        "contextual_precision": ctx_precision["score"],
        "contextual_precision_reason": ctx_precision.get("reason", ""),
        "k": k
    }


# ============================================================
# DeepEval Contextual Relevancy (Pipeline Level)
# ============================================================

def _fallback_context_relevance(
    question: str,
    retrieved_contexts: List[str]
) -> Tuple[float, str]:
    if not question or not retrieved_contexts:
        return 0.0, "Empty question or retrieved contexts."

    q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 2]
    if not q_words:
        return 0.75, "Short question; contexts treated as relevant."

    hits = 0
    for ctx in retrieved_contexts:
        ctx_lower = ctx.lower()
        if any(w in ctx_lower for w in q_words):
            hits += 1

    ratio = hits / len(retrieved_contexts)
    if ratio >= 0.8:
        return 1.0, f"High context relevancy ({hits}/{len(retrieved_contexts)} passages contain query entities)."
    elif ratio >= 0.5:
        return 0.8, f"Moderate context relevancy ({hits}/{len(retrieved_contexts)} passages contain query entities)."
    elif ratio > 0.0:
        return round(max(0.4, ratio), 4), f"Partial context relevancy ({hits}/{len(retrieved_contexts)} passages contain query entities)."
    return 0.0, "No query entities found in retrieved contexts."


def compute_context_relevance(
    question: str,
    retrieved_contexts: List[str],
    model: Any,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Measures Contextual Relevancy via DeepEval:
    Evaluates whether the retrieved context passages are strictly relevant to the question.
    """
    contexts = retrieved_contexts if retrieved_contexts else ["No context retrieved."]
    test_case = LLMTestCase(
        input=question,
        actual_output="",
        retrieval_context=contexts
    )

    metric = ContextualRelevancyMetric(threshold=threshold, model=model, include_reason=True)
    try:
        metric.measure(test_case)
        score = round(float(metric.score), 4)
        reason = metric.reason or "Retrieved contexts are relevant to the query."
        return {"score": score, "reason": reason, "success": score >= threshold}
    except Exception as e:
        fb_score, fb_reason = _fallback_context_relevance(question, contexts)
        return {"score": fb_score, "reason": f"[heuristic fallback: judge failed] {fb_reason} (DeepEval notice: {str(e)})", "success": fb_score >= threshold, "fallback": True}

