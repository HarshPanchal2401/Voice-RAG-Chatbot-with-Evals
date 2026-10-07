"""
Statistics helpers: bootstrap confidence intervals, paired comparisons, latency.
================================================================================
* ``mean_ci``          - mean with a percentile-bootstrap 95% CI. When ``groups`` are
                         given (e.g. query_id, so the Gujarati and Hindi versions of the
                         same question stay together) a *cluster* bootstrap is used.
* ``paired_bootstrap`` - mean of (b - a) over paired items, bootstrap CI and a two-sided
                         bootstrap p-value. Use it for A/B comparisons (rerank vs not).
* ``sign_test``        - exact two-sided binomial sign test on paired items (ties dropped).
* ``latency_summary``  - p50 / p95 / mean excluding warm-up requests + cold-start value.
None / NaN values are always excluded (never treated as 0).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BOOT = 2000
DEFAULT_SEED = 12345


def _valid(x: Any) -> bool:
    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and not math.isnan(float(x))


def clean_values(values: Sequence[Any]) -> List[float]:
    return [float(v) for v in values if _valid(v)]


def _percentile_sorted(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def percentile(values: Sequence[Any], q: float) -> Optional[float]:
    vals = sorted(clean_values(values))
    if not vals:
        return None
    return _percentile_sorted(vals, q)


def _cluster(values: Sequence[Any], groups: Optional[Sequence[Any]]) -> List[Tuple[float, int]]:
    """[(sum, count)] per cluster (each item is its own cluster when groups is None)."""
    if groups is None:
        return [(float(v), 1) for v in values if _valid(v)]
    sums: Dict[Any, List[float]] = {}
    for v, g in zip(values, groups):
        if _valid(v):
            acc = sums.setdefault(g, [0.0, 0])
            acc[0] += float(v)
            acc[1] += 1
    return [(s, int(c)) for s, c in sums.values()]


def _boot_means(clusters: List[Tuple[float, int]], n_boot: int, seed: int) -> List[float]:
    """Sorted bootstrap distribution of the (cluster-)resampled mean. Deterministic per seed."""
    m = len(clusters)
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        sums = np.array([c[0] for c in clusters], dtype=float)
        counts = np.array([c[1] for c in clusters], dtype=float)
        out: List[float] = []
        batch = max(1, min(n_boot, 2_000_000 // max(1, m)))
        done = 0
        while done < n_boot:
            b = min(batch, n_boot - done)
            idx = rng.integers(0, m, size=(b, m))
            out.extend((sums[idx].sum(axis=1) / counts[idx].sum(axis=1)).tolist())
            done += b
        out.sort()
        return out
    except ImportError:  # pragma: no cover - numpy is a hard dependency of the project
        rng2 = random.Random(seed)
        stats = []
        for _ in range(n_boot):
            s = 0.0
            c = 0
            for _j in range(m):
                cs, cc = clusters[rng2.randrange(m)]
                s += cs
                c += cc
            stats.append(s / c)
        stats.sort()
        return stats


def bootstrap_ci(values: Sequence[Any], groups: Optional[Sequence[Any]] = None, n_boot: int = DEFAULT_BOOT,
                 alpha: float = 0.05, seed: int = DEFAULT_SEED) -> Tuple[Optional[float], Optional[float]]:
    clusters = _cluster(values, groups)
    if len(clusters) < 2:
        return None, None
    stats = _boot_means(clusters, n_boot, seed)
    return _percentile_sorted(stats, 100 * alpha / 2), _percentile_sorted(stats, 100 * (1 - alpha / 2))


def mean_ci(values: Sequence[Any], groups: Optional[Sequence[Any]] = None, n_boot: int = DEFAULT_BOOT,
            alpha: float = 0.05, seed: int = DEFAULT_SEED) -> Dict[str, Optional[float]]:
    vals = clean_values(values)
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    lo, hi = bootstrap_ci(values, groups, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "ci_low": None if lo is None else round(lo, 4),
        "ci_high": None if hi is None else round(hi, 4),
        "n": len(vals),
    }


def paired_bootstrap(a: Sequence[Any], b: Sequence[Any], groups: Optional[Sequence[Any]] = None,
                     n_boot: int = DEFAULT_BOOT, alpha: float = 0.05, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Paired comparison of arm B against arm A on the same items: diff = b - a."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diffs, grp = [], []
    for i, (x, y) in enumerate(zip(a, b)):
        if _valid(x) and _valid(y):
            diffs.append(float(y) - float(x))
            grp.append(groups[i] if groups is not None else i)
    n = len(diffs)
    if n == 0:
        return {"n_pairs": 0, "mean_diff": None, "ci_low": None, "ci_high": None, "p_value": None}
    mean_diff = sum(diffs) / n
    clusters = _cluster(diffs, grp)
    if len(clusters) < 2:
        return {"n_pairs": n, "mean_diff": round(mean_diff, 4), "ci_low": None, "ci_high": None, "p_value": None}
    stats = _boot_means(clusters, n_boot, seed)
    le0 = sum(1 for s in stats if s <= 0) / n_boot
    ge0 = sum(1 for s in stats if s >= 0) / n_boot
    return {
        "n_pairs": n,
        "mean_diff": round(mean_diff, 4),
        "ci_low": round(_percentile_sorted(stats, 100 * alpha / 2), 4),
        "ci_high": round(_percentile_sorted(stats, 100 * (1 - alpha / 2)), 4),
        "p_value": round(min(1.0, 2 * min(le0, ge0)), 4),
    }


def sign_test(a: Sequence[Any], b: Sequence[Any]) -> Dict[str, Any]:
    """Exact two-sided sign test: how often B beats A on paired items (ties dropped)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    wins_b = wins_a = ties = 0
    for x, y in zip(a, b):
        if not (_valid(x) and _valid(y)):
            continue
        if float(y) > float(x):
            wins_b += 1
        elif float(y) < float(x):
            wins_a += 1
        else:
            ties += 1
    n = wins_a + wins_b
    if n == 0:
        p = 1.0
    else:
        k = min(wins_a, wins_b)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"wins_b": wins_b, "wins_a": wins_a, "ties": ties, "p_value": round(p, 4)}


def latency_summary(values_in_order: Sequence[Any], warmup: int = 1) -> Dict[str, Any]:
    """p50/p95/mean of a latency series excluding the first `warmup` requests (cold start)."""
    vals = [float(v) for v in values_in_order if _valid(v)]
    if not vals:
        return {"n": 0, "cold_start_ms": None, "p50_ms": None, "p95_ms": None, "mean_ms": None, "warmup_excluded": 0}
    warm = max(0, min(warmup, len(vals) - 1))
    cold = vals[0] if warm > 0 else None
    rest = sorted(vals[warm:])
    return {
        "n": len(rest),
        "cold_start_ms": None if cold is None else round(cold, 2),
        "p50_ms": round(_percentile_sorted(rest, 50), 2),
        "p95_ms": round(_percentile_sorted(rest, 95), 2),
        "mean_ms": round(sum(rest) / len(rest), 2),
        "warmup_excluded": warm,
    }
