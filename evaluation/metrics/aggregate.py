"""
Aggregation of per-query evaluation records
===========================================
Input: a list of per-query records produced by the evaluators::

    {"query_id", "language", "query_type", "status": "ok" | "error", "error",
     "scores": {metric: float | None}, "failed_metrics": [...], "skipped_metrics": {...}}

Output per metric (overall and per breakdown group)::

    {"mean", "ci_low", "ci_high",            # 95% bootstrap CI, clustered by query_id
     "n_used", "n_failed", "n_skipped", "n_records",
     "judge_success_rate"}                   # n_used / (n_used + n_failed)

Error records (pipeline/generation crashed) are excluded from every metric and counted
separately (``n_errors``, ``error_rate``, ``errors``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from evaluation.metrics.stats import DEFAULT_BOOT, mean_ci


def _metric_names(records: Iterable[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for r in records:
        for k in (r.get("scores") or {}):
            if k not in names:
                names.append(k)
    return names


def summarize_metric(records: Sequence[Dict[str, Any]], metric: str, cluster_key: Optional[str] = "query_id",
                     n_boot: int = DEFAULT_BOOT, seed: int = 12345) -> Dict[str, Any]:
    values, groups = [], []
    n_failed = n_skipped = 0
    for r in records:
        scores = r.get("scores") or {}
        v = scores.get(metric)
        if metric in (r.get("failed_metrics") or []):
            n_failed += 1
        elif metric in (r.get("skipped_metrics") or {}):
            n_skipped += 1
        values.append(v)
        groups.append(r.get(cluster_key) if cluster_key else None)
    stats = mean_ci(values, groups if cluster_key else None, n_boot=n_boot, seed=seed)
    attempted = stats["n"] + n_failed
    return {
        "mean": stats["mean"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "n_used": stats["n"],
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "n_records": len(records),
        "judge_success_rate": round(stats["n"] / attempted, 4) if attempted else None,
    }


def aggregate_records(
    records: Sequence[Dict[str, Any]],
    metric_names: Optional[Sequence[str]] = None,
    breakdowns: Sequence[str] = ("query_type", "language"),
    cluster_key: str = "query_id",
    n_boot: int = DEFAULT_BOOT,
    seed: int = 12345,
) -> Dict[str, Any]:
    ok = [r for r in records if r.get("status", "ok") == "ok"]
    errs = [r for r in records if r.get("status", "ok") != "ok"]
    names = list(metric_names) if metric_names else _metric_names(ok)
    out: Dict[str, Any] = {
        "n_records": len(records),
        "n_ok": len(ok),
        "n_errors": len(errs),
        "error_rate": round(len(errs) / len(records), 4) if records else None,
        "errors": [
            {"query_id": r.get("query_id"), "language": r.get("language"), "error": str(r.get("error"))[:500]}
            for r in errs
        ],
        "distinct_query_ids": len({r.get("query_id") for r in ok}),
        "metrics": {m: summarize_metric(ok, m, cluster_key, n_boot, seed) for m in names},
    }
    for key in breakdowns:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in ok:
            groups.setdefault(str(r.get(key, "UNKNOWN")), []).append(r)
        out[f"breakdown_by_{key}"] = {
            g: {"count": len(rs), "metrics": {m: summarize_metric(rs, m, cluster_key, n_boot, seed) for m in names}}
            for g, rs in sorted(groups.items())
        }
    return out
