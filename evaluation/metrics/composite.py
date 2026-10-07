"""
The ONE documented composite score (used offline and online)
============================================================
composite = sum(w_m * score_m for m in basis) / sum(w_m for m in basis)

Weights (fixed, renormalised over the metrics of the component's basis):

    answer_correctness  0.30
    faithfulness        0.25
    answer_relevance    0.15
    completeness        0.10
    contextual_recall   0.10
    context_relevance   0.10

Bases (which metrics a component's composite uses):

    pipeline       : faithfulness, answer_relevance, context_relevance
    application    : answer_correctness, completeness
    online_golden  : answer_correctness, faithfulness, answer_relevance, contextual_recall
    online_open    : faithfulness, answer_relevance

Rules:
* If ANY metric of the basis is None for a record (judge failure or not applicable),
  that record's composite is None (excluded from the mean, counted in the report).
  A composite is never computed from a partial basis.
* Toxicity is NOT in any composite; it is a separate pass/fail gate.
* Composites of different bases are not comparable with each other.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

COMPOSITE_WEIGHTS: Dict[str, float] = {
    "answer_correctness": 0.30,
    "faithfulness": 0.25,
    "answer_relevance": 0.15,
    "completeness": 0.10,
    "contextual_recall": 0.10,
    "context_relevance": 0.10,
}

COMPOSITE_BASES: Dict[str, Sequence[str]] = {
    "pipeline": ("faithfulness", "answer_relevance", "context_relevance"),
    "application": ("answer_correctness", "completeness"),
    "online_golden": ("answer_correctness", "faithfulness", "answer_relevance", "contextual_recall"),
    "online_open": ("faithfulness", "answer_relevance"),
}


def composite_score(scores: Mapping[str, Optional[float]], basis: Sequence[str] | str) -> Optional[float]:
    names = COMPOSITE_BASES[basis] if isinstance(basis, str) else tuple(basis)
    total_w = 0.0
    acc = 0.0
    for name in names:
        val = scores.get(name)
        if val is None:
            return None
        w = COMPOSITE_WEIGHTS[name]
        acc += w * float(val)
        total_w += w
    if total_w == 0:
        return None
    return round(acc / total_w, 4)


def composite_description(basis: str) -> Dict[str, object]:
    names = COMPOSITE_BASES[basis]
    tot = sum(COMPOSITE_WEIGHTS[n] for n in names)
    return {"basis": basis, "weights": {n: round(COMPOSITE_WEIGHTS[n] / tot, 4) for n in names},
            "rule": "None if any basis metric is None; toxicity excluded"}
