"""
Pipeline Evaluator (end-to-end RAG triad)
=========================================
Runs the LIVE production pipeline (``RAGClient.generate`` -> same ``pipeline.ask`` the API
serves) once per record, then scores:

* judge: faithfulness, answer_relevance (refusal -> 0 by rule), context_relevance
* exact: hit_rate / recall_at_k / mrr / ndcg on doc keys (did retrieval find the gold passage?)
* lexical: token_f1 vs the reference answer
* refusal_rate, refused_with_gold_retrieved
* composite_score (``evaluation.metrics.composite``, basis "pipeline"; None when any
  basis metric is None)

Pipeline failures are error records (never judged); latency is reported as p50/p95
excluding warm-up, plus the cold-start request.
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
from evaluation.metrics.base import collect, outcome, skipped
from evaluation.metrics.composite import composite_description, composite_score
from evaluation.metrics.generation_metrics import a_compute_answer_relevance, a_compute_faithfulness, refusal_outcomes
from evaluation.metrics.retrieval_metrics import a_compute_context_relevance, ir_metrics
from evaluation.metrics.text_utils import is_refusal, token_f1

PIPELINE_LLM_METRICS = ("faithfulness", "answer_relevance", "context_relevance")


class PipelineEvaluator:
    def __init__(self, client: Any = None, config: Optional[EvaluationConfig] = None,
                 llm_metrics: Sequence[str] = PIPELINE_LLM_METRICS, use_reranker: bool = False,
                 judge: Any = None, show_progress: bool = True, router: Any = None):
        self.config = config or EvaluationConfig()
        self.client = client
        self.llm_metrics = tuple(llm_metrics)
        self.use_reranker = use_reranker
        self.show_progress = show_progress
        self.judge = judge if judge is not None else (make_judge(self.config) if self.llm_metrics else None)

    async def _score(self, item: Dict[str, Any], k: int) -> Dict[str, Any]:
        record = item["record"]
        if item["error"] is not None:
            return error_record(record, item["error"], item["stage"])
        gen = item["gen"]
        q, answer, ctx = record["question"], gen["answer"], gen["contexts"]
        gold = gold_doc_keys(record)
        refused = is_refusal(answer, gen.get("no_answer"))
        th = self.config.thresholds
        res: Dict[str, Dict[str, Any]] = {}
        names, coros = [], []
        if "faithfulness" in self.llm_metrics:
            names.append("faithfulness")
            coros.append(a_compute_faithfulness(q, answer, ctx, self.judge, th.get("faithfulness", 0.7)))
        if "answer_relevance" in self.llm_metrics:
            names.append("answer_relevance")
            coros.append(a_compute_answer_relevance(q, answer, self.judge, th.get("answer_relevance", 0.7), refused=refused))
        if "context_relevance" in self.llm_metrics:
            names.append("context_relevance")
            coros.append(a_compute_context_relevance(q, ctx, self.judge, th.get("context_relevance", 0.5)))
        for n, r in zip(names, await asyncio.gather(*coros)):
            res[n] = r
        for n, v in ir_metrics(gen["doc_keys"], gold, k).items():
            res[n] = outcome(v) if v is not None else skipped("no gold passages")
        f1 = token_f1(answer, reference_answer(record))["f1"]
        res["token_f1"] = outcome(f1) if f1 is not None else skipped("no reference answer")
        res.update(refusal_outcomes(answer, gen.get("no_answer"), gold_hit(gen["doc_keys"], gold)))
        flat = collect(res)
        comp = composite_score(flat["scores"], "pipeline")
        flat["scores"]["composite_score"] = comp
        if comp is None and any(m in flat["failed_metrics"] for m in ("faithfulness", "answer_relevance", "context_relevance")):
            flat["failed_metrics"].append("composite_score")
        out = base_record(record)
        out.update({
            "status": "ok",
            "generated_answer": answer,
            "ground_truth_answer": reference_answer(record),
            "generator_model": gen.get("generator_model"),
            "refused": refused,
            "retrieved_doc_keys": gen["doc_keys"],
            "gold_doc_keys": gold,
            "retrieved_contexts_count": len(ctx),
            "timings": gen.get("timings") or {},
            "trace_id": gen.get("trace_id"),
            "warning": gen.get("warning"),
            **flat,
        })
        return out

    async def a_evaluate_dataset(self, records: List[Dict[str, Any]], k: Optional[int] = None,
                                 items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        k = k or self.config.default_k
        if items is None:
            items = collect_generations(self.client, self.config, records, k, self.use_reranker, self.show_progress)
        results = await run_concurrently(items, lambda it: self._score(it, k), self.config.concurrency,
                                         desc="Scoring pipeline", show_progress=self.show_progress)
        names = ["composite_score", *self.llm_metrics, "hit_rate", "recall_at_k", "mrr", "ndcg", "token_f1",
                 "refusal_rate", "refused_with_gold_retrieved"]
        return {
            "component": "pipeline",
            "k": k,
            "use_reranker": self.use_reranker,
            "llm_metrics": list(self.llm_metrics),
            "composite": composite_description("pipeline"),
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
