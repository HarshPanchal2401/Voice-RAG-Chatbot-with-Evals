"""
Metric outcome + failure policy (single policy for offline and online evaluation)
================================================================================
Every metric function returns an *outcome* dict::

    {"score": float | None, "reason": str, "error": str | None, "skipped": str | None}

Policy:
* A judge call that fails (API error after retries, invalid/truncated JSON, DeepEval
  error) gives ``score=None`` and an ``error`` message. The metric name is added to the
  record's ``failed_metrics``. None scores are EXCLUDED from means; reports show the
  judge success rate and the n actually used. No heuristic value is ever substituted.
* ``skipped`` (with score None) means the metric does not apply (e.g. no ground truth);
  skipped metrics are not failures.
* Heuristic / lexical scores exist only as separately named metrics
  (``lexical_*``, ``token_f1``, ...).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def outcome(score: Optional[float], reason: str = "", error: Optional[str] = None,
            skipped: Optional[str] = None) -> Dict[str, Any]:
    if score is not None:
        score = round(float(score), 4)
    return {"score": score, "reason": reason or "", "error": error, "skipped": skipped}


def failed(error: str, reason: str = "") -> Dict[str, Any]:
    return outcome(None, reason=reason, error=error or "unknown judge failure")


def skipped(why: str) -> Dict[str, Any]:
    return outcome(None, reason=why, skipped=why)


def is_failed(res: Optional[Dict[str, Any]]) -> bool:
    return bool(res) and res.get("score") is None and bool(res.get("error"))


def _metric_result(metric: Any) -> Dict[str, Any]:
    err = getattr(metric, "error", None)
    score = getattr(metric, "score", None)
    if err:
        return failed(str(err))
    if score is None:
        return failed("judge returned no score")
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return failed(f"judge returned a non-numeric score: {score!r}")
    if score_f != score_f:  # NaN
        return failed("judge returned NaN")
    return outcome(score_f, reason=getattr(metric, "reason", None) or "")


def run_metric(metric: Any, test_case: Any) -> Dict[str, Any]:
    """Synchronously measure a DeepEval metric under the failure policy."""
    try:
        metric.measure(test_case, _show_indicator=False)
    except Exception as e:  # noqa: BLE001 - any failure is a judge failure
        return failed(f"{type(e).__name__}: {e}")
    return _metric_result(metric)


async def a_run_metric(metric: Any, test_case: Any) -> Dict[str, Any]:
    """Asynchronously measure a DeepEval metric under the failure policy."""
    try:
        await metric.a_measure(test_case, _show_indicator=False)
    except Exception as e:  # noqa: BLE001
        return failed(f"{type(e).__name__}: {e}")
    return _metric_result(metric)


def collect(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten {metric: outcome} into scores / reasons / failed / skipped."""
    scores: Dict[str, Optional[float]] = {}
    reasons: Dict[str, str] = {}
    failed_metrics: List[str] = []
    skipped_metrics: Dict[str, str] = {}
    for name, res in results.items():
        scores[name] = res.get("score")
        if res.get("reason"):
            reasons[name] = res["reason"]
        if res.get("error"):
            failed_metrics.append(name)
            reasons[name] = f"[judge failed] {res['error']}"
        elif res.get("skipped"):
            skipped_metrics[name] = res["skipped"]
    return {"scores": scores, "reasons": reasons, "failed_metrics": failed_metrics, "skipped_metrics": skipped_metrics}


def first_reason(reasons: Dict[str, str], order: Iterable[str]) -> str:
    return " | ".join(f"{k}: {reasons[k]}" for k in order if reasons.get(k))
