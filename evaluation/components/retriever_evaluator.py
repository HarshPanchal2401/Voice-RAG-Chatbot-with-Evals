"""
Retriever Component Evaluator
=============================
Evaluates the RAG Retrieval Component exclusively on:
- Contextual Recall (DeepEval): Does the retrieved context contain all key facts required by the expected answer?
- Contextual Precision (DeepEval): Are the relevant passages ranked above irrelevant ones in the retrieved context?
"""

import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from evaluation.config import EvaluationConfig
from evaluation.client import RAGClient
from evaluation.metrics.retrieval_metrics import evaluate_retrieval_record
from services.evaluation_service import GroqDeepEvalModel
from core.tracing import traceable


class RetrieverEvaluator:
    """Evaluates RAG retrieval solely on Contextual Recall and Contextual Precision."""

    def __init__(
        self,
        router: Optional[Any] = None,
        client: Optional[RAGClient] = None,
        config: Optional[EvaluationConfig] = None,
        run_llm_metrics: bool = True,
        use_reranker: bool = False
    ):
        self.config = config or EvaluationConfig()
        self.client = client or RAGClient(fallback_router=router)
        self.run_llm_metrics = run_llm_metrics
        self.use_reranker = use_reranker

        # LLM judge for Contextual Recall & Precision
        self.judge_model = None
        if self.run_llm_metrics:
            self.judge_model = GroqDeepEvalModel(
                model_name=self.config.judge_model_name,
                api_key=self.config.groq_api_key
            )

        self.reranker = None
        if self.use_reranker:
            from pipeline.reranker import RAGReranker
            self.reranker = RAGReranker(
                model_name=self.config.judge_model_name,
                api_key=self.config.groq_api_key
            )

    @traceable(run_type="chain", name="eval_retriever_query")
    def evaluate_query(
        self,
        question: str,
        gt_contexts: List[str],
        gt_passage_ids: List[int],
        expected_output: str,
        lang: str = "gu",
        k: int = 5,
        precomputed_contexts: Optional[List[str]] = None,
        precomputed_pids: Optional[List[int]] = None,
        precomputed_timings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs or uses pre-retrieved chunks and measures Contextual Recall and Precision."""
        # 1. Fetch candidate chunks (with optional Re-ranking)
        if self.use_reranker and self.reranker:
            if self.client.is_server_online():
                t0 = time.perf_counter()
                fetch_k = max(10, k * 2)
                ret_res = self.client.retrieve(question, language=lang, top_k=fetch_k)
                ret_ms = (time.perf_counter() - t0) * 1000
                cand_docs = ret_res.get("documents", [])
                timings = ret_res.get("timings", {})
            elif precomputed_contexts:
                cand_docs = [
                    {"text": t, "passage_id": p}
                    for t, p in zip(precomputed_contexts, precomputed_pids or [0] * len(precomputed_contexts))
                ]
                timings = dict(precomputed_timings or {})
                ret_ms = timings.get("retrieval_total_ms", 0.0)
            else:
                t0 = time.perf_counter()
                ret_res = self.client.retrieve(question, language=lang, top_k=k)
                ret_ms = (time.perf_counter() - t0) * 1000
                cand_docs = ret_res.get("documents", [])
                timings = ret_res.get("timings", {})

            # Run Reranker on candidates
            reranked_docs, rerank_ms = self.reranker.rerank(question, cand_docs, top_k=k)
            retrieved_texts = [d.get("text", "") for d in reranked_docs]
            retrieved_pids = [d.get("passage_id", d.get("chunk_id", 0)) for d in reranked_docs]
            timings["rerank_ms"] = rerank_ms
            ret_ms += rerank_ms
        elif precomputed_contexts is not None and len(precomputed_contexts) > 0:
            retrieved_texts = precomputed_contexts[:k]
            retrieved_pids = (precomputed_pids or [])[:k]
            timings = precomputed_timings or {}
            ret_ms = timings.get("retrieval_total_ms", 0.0)
        else:
            t0 = time.perf_counter()
            ret_res = self.client.retrieve(question, language=lang, top_k=k)
            ret_ms = (time.perf_counter() - t0) * 1000

            retrieved_docs = ret_res.get("documents", [])
            retrieved_texts = [d.get("text", "") for d in retrieved_docs]
            retrieved_pids = [d.get("passage_id", d.get("chunk_id", 0)) for d in retrieved_docs]
            timings = ret_res.get("timings", {})

        # 2. Evaluate Contextual Recall and Contextual Precision
        metric_res = evaluate_retrieval_record(
            question=question,
            retrieved_contexts=retrieved_texts,
            retrieved_passage_ids=retrieved_pids,
            gt_contexts=gt_contexts,
            gt_passage_ids=gt_passage_ids,
            expected_output=expected_output,
            model=self.judge_model if self.run_llm_metrics else None,
            k=k,
            run_llm_metrics=self.run_llm_metrics
        )

        return {
            "query": question,
            "language": lang,
            "retrieved_count": len(retrieved_texts),
            "retrieved_passage_ids": retrieved_pids,
            "ground_truth_passage_ids": gt_passage_ids,
            "timings": timings,
            "retrieval_total_ms": ret_ms,
            "metrics": metric_res
        }

    def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        k: int = 5,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Evaluates an entire dataset of queries through the retriever on Contextual Recall and Precision."""
        results = []
        iterator = tqdm(records, desc=f"Evaluating Retriever (k={k})") if show_progress else records

        for record in iterator:
            question = record.get("question", "")
            gt_contexts = record.get("ground_truth_contexts", [])
            gt_pids = record.get("ground_truth_passage_ids", [])
            expected = record.get("ground_truth_answer", record.get("expected_output", ""))
            lang = record.get("language", "gu")
            cached = self.config.use_cached
            pre_contexts = record.get("retrieved_contexts") if cached else None
            pre_pids = record.get("retrieved_passage_ids") if cached else None
            pre_timings = record.get("retrieval_timings") if cached else None

            try:
                res = self.evaluate_query(
                    question=question,
                    gt_contexts=gt_contexts,
                    gt_passage_ids=gt_pids,
                    expected_output=expected,
                    lang=lang,
                    k=k,
                    precomputed_contexts=pre_contexts,
                    precomputed_pids=pre_pids,
                    precomputed_timings=pre_timings
                )
                res["query_id"] = record.get("query_id")
                res["query_type"] = record.get("query_type", "UNKNOWN")
                results.append(res)
            except Exception as e:
                print(f"⚠️ Error evaluating query '{question[:30]}...': {e}")

        # Compute aggregate averages for Contextual Recall and Precision
        aggregates = self._aggregate_results(results)

        return {
            "component": "retriever",
            "evaluated_queries": len(results),
            "k": k,
            "llm_metrics_enabled": self.run_llm_metrics,
            "aggregates": aggregates,
            "details": results
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        n = len(results)
        sums = {
            "contextual_recall": 0.0,
            "contextual_precision": 0.0,
            "retrieval_total_ms": 0.0,
        }

        by_type: Dict[str, Dict[str, float]] = {}
        by_lang: Dict[str, Dict[str, float]] = {}

        for r in results:
            m = r["metrics"]
            q_type = r.get("query_type", "UNKNOWN")
            q_lang = r.get("language", "UNKNOWN")

            sums["contextual_recall"] += m.get("contextual_recall", 0.0)
            sums["contextual_precision"] += m.get("contextual_precision", 0.0)
            sums["retrieval_total_ms"] += r.get("retrieval_total_ms", 0.0)

            # Accumulate by query_type
            if q_type not in by_type:
                by_type[q_type] = {"count": 0, "recall": 0.0, "precision": 0.0}
            by_type[q_type]["count"] += 1
            by_type[q_type]["recall"] += m.get("contextual_recall", 0.0)
            by_type[q_type]["precision"] += m.get("contextual_precision", 0.0)

            # Accumulate by language
            if q_lang not in by_lang:
                by_lang[q_lang] = {"count": 0, "recall": 0.0, "precision": 0.0}
            by_lang[q_lang]["count"] += 1
            by_lang[q_lang]["recall"] += m.get("contextual_recall", 0.0)
            by_lang[q_lang]["precision"] += m.get("contextual_precision", 0.0)

        averages = {k: round(v / n, 4) for k, v in sums.items()}

        type_breakdown = {}
        for qt, data in by_type.items():
            c = data["count"]
            type_breakdown[qt] = {
                "count": c,
                "contextual_recall": round(data["recall"] / c, 4),
                "contextual_precision": round(data["precision"] / c, 4)
            }

        lang_breakdown = {}
        for lg, data in by_lang.items():
            c = data["count"]
            lang_breakdown[lg] = {
                "count": c,
                "contextual_recall": round(data["recall"] / c, 4),
                "contextual_precision": round(data["precision"] / c, 4)
            }

        return {
            "overall_averages": averages,
            "breakdown_by_query_type": type_breakdown,
            "breakdown_by_language": lang_breakdown
        }
