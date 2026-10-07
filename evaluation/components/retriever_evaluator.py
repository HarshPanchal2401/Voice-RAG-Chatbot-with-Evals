"""
Retriever Component Evaluator
=============================
Scores retrieval on the shared doc key (``query_id:passage_id``):

* IR metrics (exact, no LLM): hit@k, recall@k, precision@k, MRR@k, nDCG@k
* optional judge metrics: contextual_recall, contextual_precision
* lexical heuristics (separately named): lexical_context_jaccard, lexical_answer_coverage

``compare_rerank=True`` runs both arms on every query against the same live index with
the same k and the same scorer (arm A: hybrid RRF order, arm B: the pipeline's own
reranker via ``use_reranker=True``) and reports a paired comparison per metric
(mean difference with bootstrap 95% CI, bootstrap p-value and sign test).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from evaluation.components.common import (
    base_record,
    error_record,
    gold_doc_keys,
    latency_block,
    make_judge,
    progress,
    reference_answer,
    run_concurrently,
)
from evaluation.config import EvaluationConfig
from evaluation.metrics.aggregate import aggregate_records
from evaluation.metrics.retrieval_metrics import (
    IR_METRICS,
    RETRIEVAL_LLM_METRICS,
    StaleSnapshotError,
    a_evaluate_retrieval_record,
    doc_keys_from_documents,
    snapshot_doc_keys,
)
from evaluation.metrics.stats import paired_bootstrap, sign_test


class RetrieverEvaluator:
    def __init__(
        self,
        client: Any = None,
        config: Optional[EvaluationConfig] = None,
        llm_metrics: Sequence[str] = RETRIEVAL_LLM_METRICS,
        use_reranker: bool = False,
        judge: Any = None,
        show_progress: bool = True,
        router: Any = None,  # deprecated, ignored (use client)
        run_llm_metrics: Optional[bool] = None,  # deprecated alias
    ):
        self.config = config or EvaluationConfig()
        if client is None:
            raise ValueError("RetrieverEvaluator needs a RAGClient (explicit --mode server|inprocess).")
        self.client = client
        if run_llm_metrics is False:
            llm_metrics = ()
        self.llm_metrics = tuple(llm_metrics)
        self.use_reranker = use_reranker
        self.show_progress = show_progress
        self.judge = judge if judge is not None else (make_judge(self.config) if self.llm_metrics else None)

    # ---------------------------------------------------------------- phase 1
    def _live(self, record: Dict[str, Any], k: int, use_reranker: bool) -> Dict[str, Any]:
        res = self.client.retrieve(record["question"], language=record["language"], top_k=k,
                                   use_reranker=use_reranker)
        docs = res.get("documents") or []
        timings = dict(res.get("timings") or {})
        timings["client_ms"] = res.get("client_ms")
        return {
            "doc_keys": doc_keys_from_documents(docs),
            "contexts": [d.get("text", "") for d in docs],
            "timings": timings,
            "original_ranks": [d.get("original_rank") for d in docs],
        }

    def _cached(self, record: Dict[str, Any], k: int) -> Dict[str, Any]:
        if record.get("retrieval_error"):
            raise RuntimeError(f"snapshot retrieval error: {record['retrieval_error']}")
        keys = snapshot_doc_keys(record)  # raises StaleSnapshotError for bare passage ids
        return {"doc_keys": keys[:k], "contexts": list(record.get("retrieved_contexts") or [])[:k],
                "timings": dict(record.get("retrieval_timings") or {}), "original_ranks": []}

    def collect(self, records: List[Dict[str, Any]], k: int, arms: Sequence[str]) -> List[Dict[str, Any]]:
        out = []
        for record in progress(records, f"Retrieving (k={k}, arms={','.join(arms)})", self.show_progress):
            item = {"record": record, "arms": {}, "error": None}
            try:
                for arm in arms:
                    if self.config.use_cached:
                        item["arms"][arm] = self._cached(record, k)
                    else:
                        item["arms"][arm] = self._live(record, k, use_reranker=(arm == "rerank"))
            except StaleSnapshotError:
                raise
            except Exception as e:  # noqa: BLE001 - pipeline errors become error records
                item["error"] = e
            out.append(item)
        return out

    # ---------------------------------------------------------------- phase 2
    async def _score(self, item: Dict[str, Any], arm: str, k: int) -> Dict[str, Any]:
        record = item["record"]
        if item["error"] is not None:
            return error_record(record, item["error"], "retrieval")
        data = item["arms"][arm]
        gold = gold_doc_keys(record)
        res = await a_evaluate_retrieval_record(
            question=record["question"],
            retrieved_contexts=data["contexts"],
            retrieved_doc_keys=data["doc_keys"],
            gold_keys=gold,
            expected_output=reference_answer(record),
            model=self.judge,
            k=k,
            llm_metrics=self.llm_metrics,
            gt_contexts=record.get("ground_truth_contexts") or [],
            thresholds=self.config.thresholds,
        )
        out = base_record(record)
        out.update({
            "status": "ok",
            "arm": arm,
            "retrieved_doc_keys": data["doc_keys"],
            "gold_doc_keys": gold,
            "retrieved_count": len(data["doc_keys"]),
            "timings": data["timings"],
            "original_ranks": data.get("original_ranks") or [],
            **res,
        })
        return out

    async def a_evaluate_dataset(self, records: List[Dict[str, Any]], k: int = 5,
                                 compare_rerank: bool = False) -> Dict[str, Any]:
        if compare_rerank and self.config.use_cached:
            raise ValueError("--compare-rerank needs live retrieval (both arms against the same index); drop --use-cached.")
        arms = ("base", "rerank") if compare_rerank else (("rerank",) if self.use_reranker else ("base",))
        items = self.collect(records, k, arms)
        per_arm: Dict[str, List[Dict[str, Any]]] = {}
        for arm in arms:
            per_arm[arm] = await run_concurrently(
                items, lambda it, a=arm: self._score(it, a, k), self.config.concurrency,
                desc=f"Scoring retrieval [{arm}]", show_progress=self.show_progress)
        metric_names = list(IR_METRICS) + list(self.llm_metrics) + ["lexical_context_jaccard", "lexical_answer_coverage"]
        if not compare_rerank:
            arm = arms[0]
            return {
                "component": "retriever_reranked" if arm == "rerank" else "retriever",
                "k": k,
                "llm_metrics": list(self.llm_metrics),
                "use_reranker": arm == "rerank",
                "evaluated_queries": sum(1 for r in per_arm[arm] if r["status"] == "ok"),
                "aggregates": aggregate_records(per_arm[arm], metric_names, n_boot=self.config.n_boot, seed=self.config.seed),
                "latency": latency_block(per_arm[arm], self.config.warmup),
                "details": per_arm[arm],
            }
        base, rer = per_arm["base"], per_arm["rerank"]
        paired = {}
        groups = [r.get("query_id") for r in base]
        for m in metric_names:
            a = [r.get("scores", {}).get(m) if r["status"] == "ok" else None for r in base]
            b = [r.get("scores", {}).get(m) if r["status"] == "ok" else None for r in rer]
            if not any(v is not None for v in a) or not any(v is not None for v in b):
                continue
            paired[m] = {**paired_bootstrap(a, b, groups=groups, n_boot=self.config.n_boot, seed=self.config.seed),
                         "sign_test": sign_test(a, b)}
        return {
            "component": "retriever_rerank_comparison",
            "k": k,
            "llm_metrics": list(self.llm_metrics),
            "evaluated_queries": sum(1 for r in base if r["status"] == "ok"),
            "arms": {
                "no_rerank": {"aggregates": aggregate_records(base, metric_names, n_boot=self.config.n_boot, seed=self.config.seed),
                              "latency": latency_block(base, self.config.warmup)},
                "rerank": {"aggregates": aggregate_records(rer, metric_names, n_boot=self.config.n_boot, seed=self.config.seed),
                           "latency": latency_block(rer, self.config.warmup)},
            },
            "paired_comparison": {"definition": "diff = rerank - no_rerank, paired per query; 95% bootstrap CI "
                                                "clustered by query_id; sign test on per-query wins",
                                  "metrics": paired},
            "aggregates": aggregate_records(rer, metric_names, n_boot=self.config.n_boot, seed=self.config.seed),
            "details": [{"query_id": b.get("query_id"), "language": b.get("language"), "no_rerank": b, "rerank": r}
                        for b, r in zip(base, rer)],
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]], k: int = 5, compare_rerank: bool = False,
                         show_progress: Optional[bool] = None) -> Dict[str, Any]:
        if show_progress is not None:
            self.show_progress = show_progress
        return asyncio.run(self.a_evaluate_dataset(records, k=k, compare_rerank=compare_rerank))
