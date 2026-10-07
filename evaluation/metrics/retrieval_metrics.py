"""
Retrieval Component Metrics
===========================
Relevance is decided on the shared document key (see CONTRACT.md)::

    doc_key = f"{query_id}:{passage_id}"

``passage_id`` / ``chunk_id`` alone are only unique *within* a corpus query group
(27 / 40 distinct values over 301,752 chunks), so comparing bare passage ids matches
unrelated passages. Gold keys are built from ``(record.query_id, ground_truth_passage_ids)``;
retrieved keys from each document's ``doc_key`` (or ``source_query_id``/``query_id`` +
``passage_id``).

IR metrics (no LLM, exact):
    hit@k, recall@k, precision@k (divides by k), MRR@k, nDCG@k (binary relevance).
    Duplicates (several chunks of the same passage) count once: only the first
    occurrence of a relevant key earns gain, so nDCG <= 1 and precision <= 1.

LLM-judge metrics (DeepEval, failure policy in ``evaluation.metrics.base``):
    contextual_recall, contextual_precision, context_relevance.

Lexical heuristics (separately named, never substituted for a judge score):
    lexical_context_jaccard, lexical_answer_coverage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from evaluation.metrics.base import a_run_metric, collect, failed, outcome, run_metric, skipped
from evaluation.metrics.text_utils import token_coverage, token_jaccard

NO_CONTEXT = "(no context retrieved)"


class MissingDocKeyError(ValueError):
    """A retrieved document carries no usable (query_id, passage_id) / doc_key."""


class StaleSnapshotError(ValueError):
    """A cached snapshot lacks doc keys (bare passage ids are not unique)."""


# ============================================================
# Document keys
# ============================================================

def _norm_id(value: Any) -> str:
    if isinstance(value, bool):
        raise MissingDocKeyError(f"invalid id {value!r}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = str(value).strip()
    if s.lstrip("-").isdigit():
        return str(int(s))
    return s


def make_doc_key(query_id: Any, passage_id: Any) -> str:
    if query_id is None or passage_id is None or str(query_id).strip() == "" or str(passage_id).strip() == "":
        raise MissingDocKeyError(f"cannot build doc_key from query_id={query_id!r}, passage_id={passage_id!r}")
    return f"{_norm_id(query_id)}:{_norm_id(passage_id)}"


def dedupe(keys: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def gold_doc_keys(record: Dict[str, Any]) -> List[str]:
    """Gold keys of a golden record (deduplicated, order kept)."""
    if record.get("ground_truth_doc_keys"):
        return dedupe(str(k) for k in record["ground_truth_doc_keys"])
    qid = record.get("query_id")
    pids = record.get("ground_truth_passage_ids") or []
    return dedupe(make_doc_key(qid, pid) for pid in pids)


def doc_key_of(doc: Dict[str, Any]) -> Optional[str]:
    """doc_key of an in-process document or an API SourceDocument (None if impossible)."""
    if not isinstance(doc, dict):
        return None
    key = doc.get("doc_key")
    if key:
        return str(key)
    qid = doc.get("source_query_id")
    if qid is None:
        qid = doc.get("query_id")
    pid = doc.get("passage_id")
    if qid is None or pid is None:
        return None
    try:
        return make_doc_key(qid, pid)
    except MissingDocKeyError:
        return None


def doc_keys_from_documents(docs: Sequence[Dict[str, Any]]) -> List[str]:
    """Retrieved keys in rank order. Raises MissingDocKeyError if any doc lacks them."""
    keys = []
    for i, d in enumerate(docs or []):
        k = doc_key_of(d)
        if k is None:
            raise MissingDocKeyError(
                f"retrieved document #{i + 1} has no doc_key / (query_id|source_query_id, passage_id); "
                "bare passage ids are not unique and cannot be scored."
            )
        keys.append(k)
    return keys


def snapshot_doc_keys(record: Dict[str, Any]) -> List[str]:
    """Doc keys stored in a cached snapshot record, or StaleSnapshotError."""
    keys = record.get("retrieved_doc_keys")
    if keys is None:
        raise StaleSnapshotError(
            "Snapshot record has no 'retrieved_doc_keys' (only bare 'retrieved_passage_ids', which are not "
            "unique across the corpus). It cannot be scored. Regenerate the snapshot with live retrieval: "
            "python -m evaluation.combine_datasets snapshot --lang <gu|hi|combined> --sample 300 --mode server"
        )
    return [str(k) for k in keys]


# ============================================================
# IR metrics on doc keys
# ============================================================

def _check_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _gains(retrieved: Sequence[Any], gold: set, k: int) -> List[int]:
    """Binary gains for the top-k positions; repeated keys earn no further gain."""
    seen = set()
    gains = []
    for key in list(retrieved)[:k]:
        if key in gold and key not in seen:
            gains.append(1)
            seen.add(key)
        else:
            gains.append(0)
    return gains


def compute_hit_rate(retrieved_keys: Sequence[Any], gold_keys: Sequence[Any], k: int = 5) -> Optional[float]:
    """1.0 if any gold key is in the top-k, else 0.0. None when there is no gold."""
    _check_k(k)
    gold = set(gold_keys or [])
    if not gold:
        return None
    return 1.0 if any(_gains(retrieved_keys or [], gold, k)) else 0.0


def compute_recall_at_k(retrieved_keys: Sequence[Any], gold_keys: Sequence[Any], k: int = 5) -> Optional[float]:
    """|unique gold keys in top-k| / |unique gold keys|."""
    _check_k(k)
    gold = set(gold_keys or [])
    if not gold:
        return None
    return round(sum(_gains(retrieved_keys or [], gold, k)) / len(gold), 4)


def compute_precision_at_k(retrieved_keys: Sequence[Any], gold_keys: Sequence[Any], k: int = 5) -> Optional[float]:
    """|unique relevant keys in top-k| / k  (always divides by k)."""
    _check_k(k)
    gold = set(gold_keys or [])
    if not gold:
        return None
    return round(sum(_gains(retrieved_keys or [], gold, k)) / k, 4)


def compute_mrr(retrieved_keys: Sequence[Any], gold_keys: Sequence[Any], k: int = 10) -> Optional[float]:
    """Reciprocal rank of the first relevant key within top-k (0 if none)."""
    _check_k(k)
    gold = set(gold_keys or [])
    if not gold:
        return None
    for rank, g in enumerate(_gains(retrieved_keys or [], gold, k), start=1):
        if g:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg(retrieved_keys: Sequence[Any], gold_keys: Sequence[Any], k: int = 10) -> Optional[float]:
    """Binary-relevance nDCG@k with duplicate keys counted once (always within [0, 1])."""
    _check_k(k)
    gold = set(gold_keys or [])
    if not gold:
        return None
    gains = _gains(retrieved_keys or [], gold, k)
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return round(dcg / idcg, 4) if idcg > 0 else None


IR_METRICS = ("hit_rate", "recall_at_k", "precision_at_k", "mrr", "ndcg")


def ir_metrics(retrieved_keys: Sequence[Any], gold_keys: Sequence[Any], k: int = 5) -> Dict[str, Optional[float]]:
    return {
        "hit_rate": compute_hit_rate(retrieved_keys, gold_keys, k),
        "recall_at_k": compute_recall_at_k(retrieved_keys, gold_keys, k),
        "precision_at_k": compute_precision_at_k(retrieved_keys, gold_keys, k),
        "mrr": compute_mrr(retrieved_keys, gold_keys, k),
        "ndcg": compute_ndcg(retrieved_keys, gold_keys, k),
    }


# ============================================================
# Lexical heuristics (separately named)
# ============================================================

def compute_context_jaccard(retrieved_contexts: List[str], gt_contexts: List[str]) -> Optional[float]:
    """Token-set Jaccard between retrieved passages and gold passages (shared tokenizer)."""
    return token_jaccard(retrieved_contexts or [], gt_contexts or [])


def compute_answer_coverage(expected_output: str, retrieved_contexts: List[str]) -> Optional[float]:
    """Fraction of distinct reference-answer tokens present in the retrieved passages."""
    return token_coverage(expected_output, retrieved_contexts or [])


# ============================================================
# LLM-judge metrics
# ============================================================

def _ctx_test_case(question: str, contexts: List[str], expected_output: Optional[str], actual_output: Optional[str] = None):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=question,
        actual_output=actual_output if actual_output is not None else (expected_output or ""),
        expected_output=expected_output,
        retrieval_context=list(contexts),
    )


def _recall_metric(model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import ContextualRecallMetric

    return ContextualRecallMetric(threshold=threshold, model=model, include_reason=True, async_mode=async_mode)


def _precision_metric(model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import ContextualPrecisionMetric

    return ContextualPrecisionMetric(threshold=threshold, model=model, include_reason=True, async_mode=async_mode)


def _relevancy_metric(model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import ContextualRelevancyMetric

    return ContextualRelevancyMetric(threshold=threshold, model=model, include_reason=True, async_mode=async_mode)


def _precheck(question: str, contexts: List[str], expected: Optional[str], model: Any, needs_expected: bool):
    if model is None:
        return failed("no judge model configured")
    if needs_expected and not (expected or "").strip():
        return skipped("no reference answer")
    if not [c for c in (contexts or []) if (c or "").strip()]:
        # Deterministic rule (not a heuristic): nothing retrieved -> nothing recalled/relevant.
        return outcome(0.0, reason="No context was retrieved.")
    return None


def compute_contextual_recall(question: str, retrieved_contexts: List[str], expected_output: str,
                              model: Any, threshold: float = 0.7, **_: Any) -> Dict[str, Any]:
    """DeepEval ContextualRecall: does the retrieved context contain the facts of the reference answer?"""
    pre = _precheck(question, retrieved_contexts, expected_output, model, True)
    if pre is not None:
        return pre
    return run_metric(_recall_metric(model, threshold, False), _ctx_test_case(question, retrieved_contexts, expected_output))


async def a_compute_contextual_recall(question: str, retrieved_contexts: List[str], expected_output: str,
                                      model: Any, threshold: float = 0.7, **_: Any) -> Dict[str, Any]:
    pre = _precheck(question, retrieved_contexts, expected_output, model, True)
    if pre is not None:
        return pre
    return await a_run_metric(_recall_metric(model, threshold, True), _ctx_test_case(question, retrieved_contexts, expected_output))


def compute_contextual_precision(question: str, retrieved_contexts: List[str], expected_output: str,
                                 model: Any, threshold: float = 0.7, **_: Any) -> Dict[str, Any]:
    """DeepEval ContextualPrecision: are relevant passages ranked above irrelevant ones?"""
    pre = _precheck(question, retrieved_contexts, expected_output, model, True)
    if pre is not None:
        return pre
    return run_metric(_precision_metric(model, threshold, False), _ctx_test_case(question, retrieved_contexts, expected_output))


async def a_compute_contextual_precision(question: str, retrieved_contexts: List[str], expected_output: str,
                                         model: Any, threshold: float = 0.7, **_: Any) -> Dict[str, Any]:
    pre = _precheck(question, retrieved_contexts, expected_output, model, True)
    if pre is not None:
        return pre
    return await a_run_metric(_precision_metric(model, threshold, True), _ctx_test_case(question, retrieved_contexts, expected_output))


def compute_context_relevance(question: str, retrieved_contexts: List[str], model: Any,
                              threshold: float = 0.5) -> Dict[str, Any]:
    """DeepEval ContextualRelevancy: are the retrieved passages relevant to the question?"""
    pre = _precheck(question, retrieved_contexts, None, model, False)
    if pre is not None:
        return pre
    return run_metric(_relevancy_metric(model, threshold, False), _ctx_test_case(question, retrieved_contexts, None, ""))


async def a_compute_context_relevance(question: str, retrieved_contexts: List[str], model: Any,
                                      threshold: float = 0.5) -> Dict[str, Any]:
    pre = _precheck(question, retrieved_contexts, None, model, False)
    if pre is not None:
        return pre
    return await a_run_metric(_relevancy_metric(model, threshold, True), _ctx_test_case(question, retrieved_contexts, None, ""))


RETRIEVAL_LLM_METRICS = ("contextual_recall", "contextual_precision")


def _ir_and_lexical(retrieved_contexts, retrieved_doc_keys, gold_keys, gt_contexts, expected_output, k):
    res: Dict[str, Dict[str, Any]] = {}
    for name, val in ir_metrics(retrieved_doc_keys, gold_keys, k).items():
        res[name] = outcome(val) if val is not None else skipped("no gold passages")
    res["lexical_context_jaccard"] = outcome(compute_context_jaccard(retrieved_contexts, gt_contexts or [])) \
        if gt_contexts else skipped("no gold contexts")
    cov = compute_answer_coverage(expected_output, retrieved_contexts)
    res["lexical_answer_coverage"] = outcome(cov) if cov is not None else skipped("no reference answer")
    return res


def evaluate_retrieval_record(
    question: str,
    retrieved_contexts: List[str],
    retrieved_doc_keys: List[str],
    gold_keys: List[str],
    expected_output: str,
    model: Optional[Any] = None,
    k: int = 5,
    llm_metrics: Sequence[str] = RETRIEVAL_LLM_METRICS,
    gt_contexts: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """IR metrics on doc keys (+ lexical heuristics) and the requested judge metrics."""
    thresholds = thresholds or {}
    res = _ir_and_lexical(retrieved_contexts, retrieved_doc_keys, gold_keys, gt_contexts, expected_output, k)
    if "contextual_recall" in llm_metrics:
        res["contextual_recall"] = compute_contextual_recall(
            question, retrieved_contexts, expected_output, model, thresholds.get("contextual_recall", 0.7))
    if "contextual_precision" in llm_metrics:
        res["contextual_precision"] = compute_contextual_precision(
            question, retrieved_contexts, expected_output, model, thresholds.get("contextual_precision", 0.7))
    out = collect(res)
    out["k"] = k
    return out


async def a_evaluate_retrieval_record(
    question: str,
    retrieved_contexts: List[str],
    retrieved_doc_keys: List[str],
    gold_keys: List[str],
    expected_output: str,
    model: Optional[Any] = None,
    k: int = 5,
    llm_metrics: Sequence[str] = RETRIEVAL_LLM_METRICS,
    gt_contexts: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    import asyncio

    thresholds = thresholds or {}
    res = _ir_and_lexical(retrieved_contexts, retrieved_doc_keys, gold_keys, gt_contexts, expected_output, k)
    names, coros = [], []
    if "contextual_recall" in llm_metrics:
        names.append("contextual_recall")
        coros.append(a_compute_contextual_recall(question, retrieved_contexts, expected_output, model,
                                                 thresholds.get("contextual_recall", 0.7)))
    if "contextual_precision" in llm_metrics:
        names.append("contextual_precision")
        coros.append(a_compute_contextual_precision(question, retrieved_contexts, expected_output, model,
                                                    thresholds.get("contextual_precision", 0.7)))
    for name, r in zip(names, await asyncio.gather(*coros)):
        res[name] = r
    out = collect(res)
    out["k"] = k
    return out
