"""
Generation Component Metrics
============================
LLM-judge metrics (DeepEval; failure policy in ``evaluation.metrics.base``):
    faithfulness       - are the answer's claims supported by the context?
    answer_relevance   - does the answer address the question?

Rule-based (deterministic, documented - not judge substitutes):
    answer_relevance of a refusal = 0.0 (a refusal does not answer the question; no
    judge call is made). ``refusal_rate`` and ``refused_with_gold_retrieved`` are
    reported so refusals are visible instead of hidden in averages.

Lexical / embedding (separately named):
    token_f1           - multiset token F1 against the reference answer (shared tokenizer)
    answer_similarity  - embedding cosine similarity (only when an embedding model is given)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from evaluation.metrics.base import a_run_metric, collect, failed, outcome, run_metric, skipped
from evaluation.metrics.text_utils import is_refusal, token_f1

NO_CONTEXT = "(no context provided)"
GENERATION_LLM_METRICS = ("faithfulness", "answer_relevance")
REFUSAL_RELEVANCE_REASON = (
    "Rule: the answer is a refusal ('information not available'), so it does not address "
    "the question; answer_relevance = 0 without a judge call."
)


def _faith_metric(model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import FaithfulnessMetric

    return FaithfulnessMetric(threshold=threshold, model=model, include_reason=True, async_mode=async_mode)


def _relevancy_metric(model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import AnswerRelevancyMetric

    return AnswerRelevancyMetric(threshold=threshold, model=model, include_reason=True, async_mode=async_mode)


def _faith_case(question: str, answer: str, contexts: List[str]):
    from deepeval.test_case import LLMTestCase

    ctx = [c for c in (contexts or []) if (c or "").strip()] or [NO_CONTEXT]
    return LLMTestCase(input=question, actual_output=answer or "", retrieval_context=ctx)


def _rel_case(question: str, answer: str):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(input=question, actual_output=answer or "")


def compute_faithfulness(question: str, answer: str, contexts: List[str], model: Any,
                         threshold: float = 0.7) -> Dict[str, Any]:
    if model is None:
        return failed("no judge model configured")
    if not (answer or "").strip():
        return skipped("empty answer")
    return run_metric(_faith_metric(model, threshold, False), _faith_case(question, answer, contexts))


async def a_compute_faithfulness(question: str, answer: str, contexts: List[str], model: Any,
                                 threshold: float = 0.7) -> Dict[str, Any]:
    if model is None:
        return failed("no judge model configured")
    if not (answer or "").strip():
        return skipped("empty answer")
    return await a_run_metric(_faith_metric(model, threshold, True), _faith_case(question, answer, contexts))


def compute_answer_relevance(question: str, answer: str, model: Any, threshold: float = 0.7,
                             refused: Optional[bool] = None) -> Dict[str, Any]:
    is_ref = refused if refused is not None else is_refusal(answer)
    if is_ref:
        return outcome(0.0, reason=REFUSAL_RELEVANCE_REASON)
    if model is None:
        return failed("no judge model configured")
    return run_metric(_relevancy_metric(model, threshold, False), _rel_case(question, answer))


async def a_compute_answer_relevance(question: str, answer: str, model: Any, threshold: float = 0.7,
                                     refused: Optional[bool] = None) -> Dict[str, Any]:
    is_ref = refused if refused is not None else is_refusal(answer)
    if is_ref:
        return outcome(0.0, reason=REFUSAL_RELEVANCE_REASON)
    if model is None:
        return failed("no judge model configured")
    return await a_run_metric(_relevancy_metric(model, threshold, True), _rel_case(question, answer))


def compute_indic_token_f1(prediction: str, ground_truth: str, lang: str = "gu") -> Dict[str, Optional[float]]:
    """Token-level precision / recall / F1 with the shared Unicode-aware tokenizer."""
    return token_f1(prediction, ground_truth)


def compute_semantic_similarity(text_a: str, text_b: str, embedding_model: Any) -> Dict[str, Any]:
    """Cosine similarity of embeddings mapped to [0, 1]. Skipped without a model; failure -> None."""
    if embedding_model is None:
        return skipped("no embedding model")
    if not text_a or not text_b:
        return skipped("empty text")
    try:
        import numpy as np

        emb_a = np.asarray(embedding_model.encode_query(text_a)[0], dtype=float)
        emb_b = np.asarray(embedding_model.encode_query(text_b)[0], dtype=float)
        denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        if denom == 0:
            return failed("zero-norm embedding")
        sim = float(np.dot(emb_a, emb_b)) / denom
        return outcome(max(0.0, min(1.0, (sim + 1.0) / 2.0)), reason="embedding cosine mapped to [0,1]")
    except Exception as e:  # noqa: BLE001
        return failed(f"embedding similarity failed: {type(e).__name__}: {e}")


def refusal_outcomes(answer: str, no_answer: Optional[bool], gold_retrieved: Optional[bool]) -> Dict[str, Dict[str, Any]]:
    refused = is_refusal(answer, no_answer)
    res = {"refusal_rate": outcome(1.0 if refused else 0.0)}
    if gold_retrieved is None:
        res["refused_with_gold_retrieved"] = skipped("gold retrieval unknown")
    else:
        res["refused_with_gold_retrieved"] = outcome(1.0 if (refused and gold_retrieved) else 0.0)
    return res


def _lexical(generated_answer, ground_truth_answer, embedding_model, no_answer, gold_retrieved):
    res: Dict[str, Dict[str, Any]] = {}
    f1 = token_f1(generated_answer, ground_truth_answer)["f1"]
    res["token_f1"] = outcome(f1) if f1 is not None else skipped("no reference answer or empty answer")
    if embedding_model is not None:
        res["answer_similarity"] = compute_semantic_similarity(generated_answer, ground_truth_answer, embedding_model)
    res.update(refusal_outcomes(generated_answer, no_answer, gold_retrieved))
    return res


def evaluate_generation_record(
    question: str,
    generated_answer: str,
    ground_truth_answer: Optional[str],
    contexts: List[str],
    model: Any,
    embedding_model: Optional[Any] = None,
    lang: str = "gu",
    llm_metrics: Sequence[str] = GENERATION_LLM_METRICS,
    no_answer: Optional[bool] = None,
    gold_retrieved: Optional[bool] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {}
    res = _lexical(generated_answer, ground_truth_answer, embedding_model, no_answer, gold_retrieved)
    refused = res["refusal_rate"]["score"] == 1.0
    if "faithfulness" in llm_metrics:
        res["faithfulness"] = compute_faithfulness(question, generated_answer, contexts, model,
                                                   thresholds.get("faithfulness", 0.7))
    if "answer_relevance" in llm_metrics:
        res["answer_relevance"] = compute_answer_relevance(question, generated_answer, model,
                                                           thresholds.get("answer_relevance", 0.7), refused=refused)
    out = collect(res)
    out["refused"] = refused
    return out


async def a_evaluate_generation_record(
    question: str,
    generated_answer: str,
    ground_truth_answer: Optional[str],
    contexts: List[str],
    model: Any,
    embedding_model: Optional[Any] = None,
    lang: str = "gu",
    llm_metrics: Sequence[str] = GENERATION_LLM_METRICS,
    no_answer: Optional[bool] = None,
    gold_retrieved: Optional[bool] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {}
    res = _lexical(generated_answer, ground_truth_answer, embedding_model, no_answer, gold_retrieved)
    refused = res["refusal_rate"]["score"] == 1.0
    names, coros = [], []
    if "faithfulness" in llm_metrics:
        names.append("faithfulness")
        coros.append(a_compute_faithfulness(question, generated_answer, contexts, model,
                                            thresholds.get("faithfulness", 0.7)))
    if "answer_relevance" in llm_metrics:
        names.append("answer_relevance")
        coros.append(a_compute_answer_relevance(question, generated_answer, model,
                                                thresholds.get("answer_relevance", 0.7), refused=refused))
    for name, r in zip(names, await asyncio.gather(*coros)):
        res[name] = r
    out = collect(res)
    out["refused"] = refused
    return out
