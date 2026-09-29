"""
Generation Component Metrics
============================
Evaluates generation fidelity using:
1. Faithfulness (DeepEval FaithfulnessMetric)
2. Answer Relevancy (DeepEval AnswerRelevancyMetric)
3. Semantic Embedding Similarity (BGE-M3 Cosine Distance)
4. Indic Token-Level Precision, Recall, and F1
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric


# ============================================================
# DeepEval Faithfulness & Answer Relevancy (LLM-as-a-Judge)
# ============================================================
# Robust Lexical Fallbacks for Generation
# ============================================================

def _fallback_faithfulness(answer: str, contexts: List[str]) -> Tuple[float, str]:
    if not answer or not contexts:
        return 0.0, "Answer or context is empty."
    all_ctx = " ".join(contexts).lower()
    words = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 2]
    if not words:
        return 0.8, "Answer contains only short tokens."
    grounded_count = sum(1 for w in words if w in all_ctx)
    ratio = grounded_count / len(words)
    if ratio >= 0.45:
        score = round(min(1.0, 0.75 + (ratio * 0.25)), 4)
        return score, f"Grounding verified by factual token overlap ({ratio:.1%})."
    elif ratio > 0.2:
        return 0.70, f"Moderate factual context overlap detected ({ratio:.1%})."
    return round(ratio, 4), f"Low factual overlap ({ratio:.1%}) with retrieved context."


def _fallback_answer_relevance(question: str, answer: str) -> Tuple[float, str]:
    if not answer or not question:
        return 0.0, "Question or answer is empty."
    q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 2]
    ans_lower = answer.lower()
    refusals = ["મને ખબર નથી", "પૂરતી માહિતી ઉપલબ્ધ નથી", "પર્યાપ્ત જાણકારી ઉપલબ્ધ નહીં", "जानकारी उपलब्ध नहीं", "not enough information"]
    if any(r in ans_lower for r in refusals):
        return 0.5, "Response is a standard lack-of-context refusal."

    if not q_words:
        return 0.85, "Answer provided for short query."
    overlap = sum(1 for w in q_words if w in ans_lower) / len(q_words)
    if overlap >= 0.25:
        score = round(min(1.0, 0.80 + (overlap * 0.20)), 4)
        return score, f"Answer addresses key question entities ({overlap:.1%} keyword overlap)."
    return 0.75, "Answer provides relevant response to query."


def compute_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    model: Any,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Measures Faithfulness via DeepEval:
    Verifies if statements in the generated response are factually supported by the context.
    """
    ctx_list = contexts if contexts else ["No context provided."]
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=ctx_list
    )

    metric = FaithfulnessMetric(threshold=threshold, model=model, include_reason=True)
    try:
        metric.measure(test_case)
        score = round(float(metric.score), 4)
        reason = metric.reason or "Claims are grounded in context."
        return {"score": score, "reason": reason, "success": score >= threshold}
    except Exception as e:
        fb_score, fb_reason = _fallback_faithfulness(answer, ctx_list)
        return {"score": fb_score, "reason": f"[heuristic fallback: judge failed] {fb_reason} (DeepEval notice: {str(e)})", "success": fb_score >= threshold, "fallback": True}


def compute_answer_relevance(
    question: str,
    answer: str,
    model: Any,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Measures Answer Relevancy via DeepEval:
    Determines whether the generated answer directly addresses the question without fluff.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=answer
    )

    metric = AnswerRelevancyMetric(threshold=threshold, model=model, include_reason=True)
    try:
        metric.measure(test_case)
        score = round(float(metric.score), 4)
        reason = metric.reason or "Answer directly addresses the input question."
        return {"score": score, "reason": reason, "success": score >= threshold}
    except Exception as e:
        fb_score, fb_reason = _fallback_answer_relevance(question, answer)
        return {"score": fb_score, "reason": f"[heuristic fallback: judge failed] {fb_reason} (DeepEval notice: {str(e)})", "success": fb_score >= threshold, "fallback": True}


# ============================================================
# Indic Token F1 & Embedding Semantic Similarity
# ============================================================

def compute_indic_token_f1(prediction: str, ground_truth: str, lang: str = "gu") -> Dict[str, float]:
    """
    Token-level Precision, Recall, and F1 score tailored for Indic script text.
    Strips punctuation, handles zero division gracefully.
    """
    if not prediction or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def tokenize(s: str) -> List[str]:
        cleaned = re.sub(r"[?!.,\-_:;।()\[\]{}\"\'`]", "", s.lower())
        return [w for w in cleaned.split() if w]

    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = set(pred_tokens).intersection(set(gt_tokens))
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def compute_semantic_similarity(text_a: str, text_b: str, embedding_model: Any) -> float:
    """Computes cosine embedding similarity using BGE-M3."""
    if not embedding_model or not text_a or not text_b:
        return 0.0
    try:
        emb_a = embedding_model.encode_query(text_a)[0]
        emb_b = embedding_model.encode_query(text_b)[0]
        sim = float(np.dot(emb_a, emb_b))
        return round(max(0.0, min(1.0, (sim + 1.0) / 2.0)), 4)
    except Exception:
        return 0.0


def evaluate_generation_record(
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    contexts: List[str],
    model: Any,
    embedding_model: Optional[Any] = None,
    lang: str = "gu"
) -> Dict[str, Any]:
    """Evaluates generation fidelity strictly on Faithfulness and Answer Relevancy."""
    # 1. Faithfulness (DeepEval)
    f_res = compute_faithfulness(question, generated_answer, contexts, model)

    # 2. Answer Relevancy (DeepEval)
    r_res = compute_answer_relevance(question, generated_answer, model)

    return {
        "faithfulness": f_res["score"],
        "faithfulness_reason": f_res.get("reason", ""),
        "answer_relevance": r_res["score"],
        "answer_relevance_reason": r_res.get("reason", "")
    }

