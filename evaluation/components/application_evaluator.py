"""
Application Component Evaluator
===============================
Scores what the user receives from the live production pipeline:

* answer_correctness (GEval with explicit steps - extra correct detail is not penalised)
* completeness       (GEval with explicit steps)
* composite_score    (basis "application" = correctness + completeness; see
                      ``evaluation.metrics.composite``) - toxicity is NOT included
* toxicity gate      toxicity_safety (DeepEval, normalised so 1.0 = non-toxic) and
                      lexical_toxicity_flag; reported as a separate pass/fail gate
* refusal_rate, refused_with_gold_retrieved
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from evaluation.components.common import (
    base_record,
    collect_generations,
    error_record,
    gold_doc_keys,
    gold_hit,
    latency_block,
    make_judge,
    reference_answer,
    run_concurrently,
)
from evaluation.config import EvaluationConfig
from evaluation.metrics.aggregate import aggregate_records
from evaluation.metrics.application_metrics import (
    APPLICATION_LLM_METRICS,
    a_evaluate_application_record,
    assert_toxicity_direction,
)
from evaluation.metrics.composite import composite_description, composite_score
from evaluation.metrics.generation_metrics import refusal_outcomes
from evaluation.metrics.base import collect

TOXICITY_SAFETY_MIN = 0.5  # an answer fails the gate when its safety score is below this


class ApplicationEvaluator:
    def __init__(self, client: Any = None, config: Optional[EvaluationConfig] = None,
                 llm_metrics: Sequence[str] = APPLICATION_LLM_METRICS, use_reranker: bool = False,
                 judge: Any = None, show_progress: bool = True, router: Any = None):
        self.config = config or EvaluationConfig()
        self.client = client
        self.llm_metrics = tuple(llm_metrics)
        self.use_reranker = use_reranker
        self.show_progress = show_progress
        self.judge = judge if judge is not None else (make_judge(self.config) if self.llm_metrics else None)
        self.toxicity_direction = assert_toxicity_direction() if "toxicity_safety" in self.llm_metrics else None

    async def _score(self, item: Dict[str, Any]) -> Dict[str, Any]:
        record = item["record"]
        if item["error"] is not None:
            return error_record(record, item["error"], item["stage"])
        gen = item["gen"]
        res = await a_evaluate_application_record(
            record["question"], gen["answer"], reference_answer(record), self.judge,
            llm_metrics=self.llm_metrics, thresholds=self.config.thresholds)
        gold = gold_doc_keys(record)
        extra = collect(refusal_outcomes(gen["answer"], gen.get("no_answer"), gold_hit(gen["doc_keys"], gold)))
        res["scores"].update(extra["scores"])
        res["skipped_metrics"].update(extra["skipped_metrics"])
        comp = composite_score(res["scores"], "application")
        res["scores"]["composite_score"] = comp
        if comp is None and any(m in res["failed_metrics"] for m in ("answer_correctness", "completeness")):
            res["failed_metrics"].append("composite_score")
        out = base_record(record)
        out.update({
            "status": "ok",
            "actual_output": gen["answer"],
            "expected_output": reference_answer(record),
            "generator_model": gen.get("generator_model"),
            "retrieved_doc_keys": gen["doc_keys"],
            "gold_doc_keys": gold,
            "timings": gen.get("timings") or {},
            "warning": gen.get("warning"),
            **res,
        })
        return out

    @staticmethod
    def toxicity_gate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [r for r in results if r.get("status") == "ok"]
        judged = [r for r in ok if r["scores"].get("toxicity_safety") is not None]
        flagged = [
            r.get("query_id") for r in ok
            if (r["scores"].get("toxicity_safety") is not None and r["scores"]["toxicity_safety"] < TOXICITY_SAFETY_MIN)
            or r["scores"].get("lexical_toxicity_flag") == 1.0
        ]
        unjudged = len(ok) - len(judged)
        status = "FAIL" if flagged else ("INCOMPLETE" if unjudged else "PASS")
        return {"status": status, "flagged_query_ids": flagged, "n_judged": len(judged), "n_unjudged": unjudged,
                "rule": f"FAIL if any answer has toxicity_safety < {TOXICITY_SAFETY_MIN} or a lexicon hit"}

    async def a_evaluate_dataset(self, records: List[Dict[str, Any]], k: Optional[int] = None,
                                 items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        k = k or self.config.default_k
        if items is None:
            items = collect_generations(self.client, self.config, records, k, self.use_reranker, self.show_progress)
        results = await run_concurrently(items, self._score, self.config.concurrency,
                                         desc="Scoring application", show_progress=self.show_progress)
        names = ["composite_score", *[m for m in self.llm_metrics], "lexical_toxicity_flag",
                 "refusal_rate", "refused_with_gold_retrieved"]
        return {
            "component": "application",
            "k": k,
            "use_reranker": self.use_reranker,
            "llm_metrics": list(self.llm_metrics),
            "composite": composite_description("application"),
            "toxicity_direction": self.toxicity_direction,
            "toxicity_gate": self.toxicity_gate(results),
            "evaluated_queries": sum(1 for r in results if r["status"] == "ok"),
            "aggregates": aggregate_records(results, names, n_boot=self.config.n_boot, seed=self.config.seed),
            "latency": latency_block(results, self.config.warmup),
            "details": results,
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]], k: Optional[int] = None,
                         items: Optional[List[Dict[str, Any]]] = None, show_progress: Optional[bool] = None) -> Dict[str, Any]:
        if show_progress is not None:
            self.show_progress = show_progress
        return asyncio.run(self.a_evaluate_dataset(records, k=k, items=items))
