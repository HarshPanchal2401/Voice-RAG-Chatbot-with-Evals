"""
Pipeline Evaluator (End-to-End RAG Triad Benchmark)
===================================================
Evaluates the complete Voice RAG pipeline strictly on:
1. Faithfulness (DeepEval): Is the answer factually grounded in the retrieved contexts?
2. Answer Relevancy (DeepEval): Does the answer directly address the user query without drift?
3. Context Relevancy (DeepEval): Are the retrieved passages relevant to the user query?
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

from evaluation.config import EvaluationConfig
from evaluation.client import RAGClient
from evaluation.components.retriever_evaluator import RetrieverEvaluator
from evaluation.components.generator_evaluator import GeneratorEvaluator
from evaluation.metrics.generation_metrics import compute_faithfulness, compute_answer_relevance
from evaluation.metrics.retrieval_metrics import compute_context_relevance
from services.evaluation_service import GroqDeepEvalModel
from core.tracing import traceable


class PipelineEvaluator:
    """End-to-end RAG Triad evaluator measuring Faithfulness, Answer Relevance, and Context Relevance."""

    def __init__(
        self,
        router: Optional[Any] = None,
        client: Optional[RAGClient] = None,
        config: Optional[EvaluationConfig] = None
    ):
        self.config = config or EvaluationConfig()
        self.client = client or RAGClient(fallback_router=router)
        self.router = router
        self.judge_model = GroqDeepEvalModel(
            model_name=self.config.judge_model_name,
            api_key=self.config.groq_api_key
        )
        self.retriever_eval = RetrieverEvaluator(
            router=self.router, client=self.client, config=self.config, run_llm_metrics=True
        )
        self.generator_eval = GeneratorEvaluator(
            router=self.router, client=self.client, config=self.config
        )

    @traceable(run_type="chain", name="eval_rag_triad")
    def evaluate_pipeline_record(
        self,
        question: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth_answer: str = "",
        lang: str = "gu",
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        generation_timings: Optional[Dict[str, Any]] = None,
        retrieval_timings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluates a single question-retrieval-answer record on the RAG Triad metrics."""
        # 1. Faithfulness (Answer -> Context)
        f_res = compute_faithfulness(
            question=question,
            answer=generated_answer,
            contexts=retrieved_contexts,
            model=self.judge_model,
            threshold=self.config.faithfulness_threshold
        )

        # 2. Answer Relevancy (Answer -> Question)
        a_res = compute_answer_relevance(
            question=question,
            answer=generated_answer,
            model=self.judge_model,
            threshold=self.config.answer_relevancy_threshold
        )

        # 3. Context Relevancy (Context -> Question)
        c_res = compute_context_relevance(
            question=question,
            retrieved_contexts=retrieved_contexts,
            model=self.judge_model,
            threshold=self.config.context_relevance_threshold
        )

        w_f = self.config.weights.get("faithfulness", 0.35)
        w_a = self.config.weights.get("answer_relevance", 0.35)
        w_c = self.config.weights.get("context_relevance", 0.30)
        total_w = w_f + w_a + w_c

        composite_score = round(
            ((f_res["score"] * w_f) + (a_res["score"] * w_a) + (c_res["score"] * w_c)) / total_w,
            4
        )

        total_latency_ms = round(retrieval_ms + generation_ms, 2)

        return {
            "query": question,
            "language": lang,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "retrieved_contexts_count": len(retrieved_contexts),
            "composite_score": composite_score,
            "metrics": {
                "faithfulness": f_res["score"],
                "faithfulness_reason": f_res.get("reason", ""),
                "answer_relevance": a_res["score"],
                "answer_relevance_reason": a_res.get("reason", ""),
                "context_relevance": c_res["score"],
                "context_relevance_reason": c_res.get("reason", "")
            },
            "timings": {
                "retrieval_total_ms": retrieval_ms,
                "llm_generation_ms": generation_ms,
                "total_pipeline_ms": total_latency_ms,
                "generation_profile": generation_timings or {},
                "retrieval_profile": retrieval_timings or {}
            }
        }

    @traceable(run_type="chain", name="eval_pipeline_query")
    def evaluate_query(
        self,
        question: str,
        gt_contexts: Optional[List[str]] = None,
        gt_passage_ids: Optional[List[int]] = None,
        gt_answer: str = "",
        lang: str = "gu",
        k: int = 5
    ) -> Dict[str, Any]:
        """Runs the LIVE production pipeline (same pipeline.ask the API serves), then scores the RAG Triad."""
        rag = self.client.generate(question, language=lang, top_k=k)
        retrieved_texts = [d.get("text", "") for d in rag.get("documents", [])]
        ret_timings = rag.get("retrieval_timings", {}) or {}

        result = self.evaluate_pipeline_record(
            question=question,
            retrieved_contexts=retrieved_texts,
            generated_answer=rag.get("answer", ""),
            ground_truth_answer=gt_answer,
            lang=lang,
            retrieval_ms=ret_timings.get("retrieval_total_ms", 0.0),
            generation_ms=rag.get("llm_ms", 0.0),
            generation_timings={"llm_ms": rag.get("llm_ms", 0.0), "ttft_ms": rag.get("ttft_ms", 0.0)},
            retrieval_timings=ret_timings
        )
        result["trace_id"] = rag.get("trace_id")
        result["timings"]["ttft_ms"] = rag.get("ttft_ms", 0.0)
        return result

    def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        k: int = 5,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Runs full RAG Triad evaluation across a dataset."""
        results = []
        iterator = tqdm(records, desc="Evaluating RAG Pipeline (Triad)") if show_progress else records

        by_type_acc = defaultdict(lambda: {"count": 0, "f": 0.0, "a": 0.0, "c": 0.0, "comp": 0.0})
        by_lang_acc = defaultdict(lambda: {"count": 0, "f": 0.0, "a": 0.0, "c": 0.0, "comp": 0.0})

        for record in iterator:
            question = record.get("question", "")
            lang = record.get("language", "gu")
            q_type = record.get("query_type", "UNKNOWN")
            gt_answer = record.get("ground_truth_answer", record.get("expected_output", ""))

            # Use pre-computed contexts and answers if present (e.g. from pipeline_golden_dataset)
            pre_contexts = record.get("retrieved_contexts") if self.config.use_cached else None
            pre_answer = record.get("generated_answer") if self.config.use_cached else None

            try:
                if pre_contexts is not None and pre_answer:
                    ret_ms = record.get("retrieval_timings", {}).get("retrieval_total_ms", 0.0)
                    gen_ms = record.get("generation_timings", {}).get("llm_ms", 0.0)
                    item = self.evaluate_pipeline_record(
                        question=question,
                        retrieved_contexts=pre_contexts[:k],
                        generated_answer=pre_answer,
                        ground_truth_answer=gt_answer,
                        lang=lang,
                        retrieval_ms=ret_ms,
                        generation_ms=gen_ms,
                        generation_timings=record.get("generation_timings"),
                        retrieval_timings=record.get("retrieval_timings")
                    )
                else:
                    item = self.evaluate_query(
                        question=question,
                        gt_contexts=record.get("ground_truth_contexts", []),
                        gt_passage_ids=record.get("ground_truth_passage_ids", []),
                        gt_answer=gt_answer,
                        lang=lang,
                        k=k
                    )

                item["query_id"] = record.get("query_id")
                item["query_type"] = q_type
                results.append(item)

                m = item["metrics"]
                for acc in (by_type_acc[q_type], by_lang_acc[lang]):
                    acc["count"] += 1
                    acc["f"] += m["faithfulness"]
                    acc["a"] += m["answer_relevance"]
                    acc["c"] += m["context_relevance"]
                    acc["comp"] += item["composite_score"]

            except Exception as e:
                print(f"⚠️ Error evaluating pipeline on '{question[:30]}...': {e}")

        n = len(results)
        if n == 0:
            return {"evaluated_queries": 0, "component": "pipeline"}

        avg_comp = round(sum(r["composite_score"] for r in results) / n, 4)
        avg_faith = round(sum(r["metrics"]["faithfulness"] for r in results) / n, 4)
        avg_relev = round(sum(r["metrics"]["answer_relevance"] for r in results) / n, 4)
        avg_ctx_rel = round(sum(r["metrics"]["context_relevance"] for r in results) / n, 4)
        avg_lat = round(sum(r["timings"]["total_pipeline_ms"] for r in results) / n, 2)
        avg_ret_lat = round(sum(r["timings"]["retrieval_total_ms"] for r in results) / n, 2)
        avg_llm_lat = round(sum(r["timings"]["llm_generation_ms"] for r in results) / n, 2)

        # Breakdowns
        by_type = {}
        for qt, d in by_type_acc.items():
            cnt = d["count"]
            by_type[qt] = {
                "count": cnt,
                "composite_score": round(d["comp"] / cnt, 4),
                "faithfulness": round(d["f"] / cnt, 4),
                "answer_relevance": round(d["a"] / cnt, 4),
                "context_relevance": round(d["c"] / cnt, 4)
            }

        by_lang = {}
        for lg, d in by_lang_acc.items():
            cnt = d["count"]
            by_lang[lg] = {
                "count": cnt,
                "composite_score": round(d["comp"] / cnt, 4),
                "faithfulness": round(d["f"] / cnt, 4),
                "answer_relevance": round(d["a"] / cnt, 4),
                "context_relevance": round(d["c"] / cnt, 4)
            }

        return {
            "component": "pipeline",
            "evaluated_queries": n,
            "k": k,
            "aggregates": {
                "overall_averages": {
                    "composite_score": avg_comp,
                    "faithfulness": avg_faith,
                    "answer_relevance": avg_relev,
                    "context_relevance": avg_ctx_rel,
                    "total_pipeline_ms": avg_lat,
                    "retrieval_total_ms": avg_ret_lat,
                    "llm_generation_ms": avg_llm_lat
                },
                "breakdown_by_query_type": by_type,
                "breakdown_by_language": by_lang
            },
            "details": results
        }
