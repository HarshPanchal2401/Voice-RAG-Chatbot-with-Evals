"""
Generator Component Evaluator
=============================
Evaluates the RAG Generation Component exclusively on:
- Faithfulness (DeepEval): Verifies that all statements in the answer are strictly supported by context.
- Answer Relevancy (DeepEval): Measures whether the generated answer directly addresses the question without drift.
- LLM Generation Latency & TTFT Profiling.
"""

import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from evaluation.config import EvaluationConfig, GROQ_MODEL_NAME
from pipeline.generator import generate_answer as production_generate
from evaluation.client import RAGClient
from evaluation.metrics.generation_metrics import evaluate_generation_record
from services.evaluation_service import GroqDeepEvalModel
from core.tracing import traceable


class GeneratorEvaluator:
    """Evaluates LLM generation exclusively on Faithfulness and Answer Relevancy."""

    def __init__(
        self,
        router: Optional[Any] = None,
        client: Optional[RAGClient] = None,
        config: Optional[EvaluationConfig] = None,
        use_reranker: bool = False
    ):
        self.config = config or EvaluationConfig()
        self.client = client or RAGClient(fallback_router=router)
        self.router = router
        self.use_reranker = use_reranker

        api_key = self.config.groq_api_key or os.environ.get("GROQ_API_KEY")

        # DeepEval judge model
        self.judge_model = GroqDeepEvalModel(
            model_name=self.config.judge_model_name,
            api_key=api_key
        )

        self.reranker = None
        if self.use_reranker:
            from pipeline.reranker import RAGReranker
            self.reranker = RAGReranker(
                model_name=self.config.judge_model_name,
                api_key=api_key
            )

    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        lang: str = "gu"
    ) -> Dict[str, Any]:
        """Generates with the exact production prompt + model (generation.py)."""
        try:
            res = production_generate(question, contexts, lang=lang)
            answer = res.get("answer", "")
        except Exception as e:
            res = {"llm_ms": 0.0, "ttft_ms": 0.0, "model": GROQ_MODEL_NAME}
            answer = f"Error generating answer: {e}"

        return {
            "answer": answer,
            "llm_ms": res.get("llm_ms", 0.0),
            "ttft_ms": res.get("ttft_ms", 0.0),
            "model": res.get("model", GROQ_MODEL_NAME),
        }

    @traceable(run_type="chain", name="eval_generator_query")
    def evaluate_query(
        self,
        question: str,
        ground_truth_answer: str,
        contexts: List[str],
        lang: str = "gu",
        oracle_mode: bool = True,
        precomputed_answer: Optional[str] = None,
        precomputed_timings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluates Faithfulness and Answer Relevancy for a single query."""
        # 1. Use pre-generated answer if present in dataset and in oracle mode, else generate live
        if precomputed_answer:
            generated_answer = precomputed_answer
            timings = precomputed_timings or {}
            llm_ms = timings.get("llm_ms", 0.0)
            ttft_ms = timings.get("ttft_ms", 0.0)
        else:
            gen_res = self.generate_answer(question=question, contexts=contexts, lang=lang)
            generated_answer = gen_res["answer"]
            llm_ms = gen_res["llm_ms"]
            ttft_ms = gen_res["ttft_ms"]

        # 2. Evaluate strictly on Faithfulness and Answer Relevancy via DeepEval
        metric_res = evaluate_generation_record(
            question=question,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth_answer,
            contexts=contexts,
            model=self.judge_model,
            lang=lang
        )

        return {
            "query": question,
            "language": lang,
            "mode": "oracle" if oracle_mode else "retrieved",
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "llm_ms": llm_ms,
            "ttft_ms": ttft_ms,
            "metrics": metric_res
        }

    def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        oracle_context: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Evaluates an entire dataset of queries on Faithfulness and Answer Relevancy."""
        results = []
        mode_desc = "Oracle" if oracle_context else ("RAG + Re-Ranker" if self.use_reranker else "RAG Standard")
        iterator = tqdm(records, desc=f"Evaluating Generator ({mode_desc} Mode)") if show_progress else records

        for record in iterator:
            question = record.get("question", "")
            gt_answer = record.get("ground_truth_answer", record.get("expected_output", ""))
            lang = record.get("language", "gu")
            pre_answer = record.get("generated_answer")
            pre_timings = record.get("generation_timings")

            # Determine contexts: gold contexts or retrieved
            if oracle_context:
                contexts = record.get("ground_truth_contexts", record.get("contexts", []))
                if not contexts and "context" in record and record["context"]:
                    contexts = [record["context"]]
                eval_answer = pre_answer if self.config.use_cached else None
                eval_timings = pre_timings if self.config.use_cached else None
            else:
                if self.use_reranker and self.reranker:
                    # Fetch candidate pool (10) from hybrid retrieval
                    ret = self.client.retrieve(question, language=lang, top_k=10)
                    cand_docs = ret.get("documents", [])
                    # Re-rank candidates down to top 5
                    reranked_docs, _ = self.reranker.rerank(question, cand_docs, top_k=5)
                    contexts = [d.get("text", "") for d in reranked_docs]
                else:
                    ret = self.client.retrieve(question, language=lang, top_k=5)
                    contexts = [d.get("text", "") for d in ret.get("documents", [])]

                # In RAG mode, generate fresh answers using the retrieved/reranked contexts
                eval_answer = None
                eval_timings = None

            try:
                res = self.evaluate_query(
                    question=question,
                    ground_truth_answer=gt_answer,
                    contexts=contexts,
                    lang=lang,
                    oracle_mode=oracle_context,
                    precomputed_answer=eval_answer,
                    precomputed_timings=eval_timings
                )
                res["query_id"] = record.get("query_id")
                res["query_type"] = record.get("query_type", "UNKNOWN")
                results.append(res)
            except Exception as e:
                print(f"⚠️ Error evaluating generator on '{question[:30]}...': {e}")

        aggregates = self._aggregate_results(results)

        return {
            "component": "generator",
            "mode": "oracle" if oracle_context else "retrieved",
            "evaluated_queries": len(results),
            "aggregates": aggregates,
            "details": results
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        n = len(results)
        sums = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "llm_ms": 0.0,
            "ttft_ms": 0.0
        }

        by_type: Dict[str, Dict[str, float]] = {}
        by_lang: Dict[str, Dict[str, float]] = {}

        for r in results:
            m = r["metrics"]
            q_type = r.get("query_type", "UNKNOWN")
            q_lang = r.get("language", "UNKNOWN")

            sums["faithfulness"] += m.get("faithfulness", 0.0)
            sums["answer_relevance"] += m.get("answer_relevance", 0.0)
            sums["llm_ms"] += r.get("llm_ms", 0.0)
            sums["ttft_ms"] += r.get("ttft_ms", 0.0)

            # Query type breakdown
            if q_type not in by_type:
                by_type[q_type] = {"count": 0, "faithfulness": 0.0, "relevance": 0.0}
            by_type[q_type]["count"] += 1
            by_type[q_type]["faithfulness"] += m.get("faithfulness", 0.0)
            by_type[q_type]["relevance"] += m.get("answer_relevance", 0.0)

            # Lang breakdown
            if q_lang not in by_lang:
                by_lang[q_lang] = {"count": 0, "faithfulness": 0.0, "relevance": 0.0}
            by_lang[q_lang]["count"] += 1
            by_lang[q_lang]["faithfulness"] += m.get("faithfulness", 0.0)
            by_lang[q_lang]["relevance"] += m.get("answer_relevance", 0.0)

        averages = {k: round(v / n, 4) for k, v in sums.items()}

        type_breakdown = {}
        for qt, data in by_type.items():
            c = data["count"]
            type_breakdown[qt] = {
                "count": c,
                "faithfulness": round(data["faithfulness"] / c, 4),
                "answer_relevance": round(data["relevance"] / c, 4)
            }

        lang_breakdown = {}
        for lg, data in by_lang.items():
            c = data["count"]
            lang_breakdown[lg] = {
                "count": c,
                "faithfulness": round(data["faithfulness"] / c, 4),
                "answer_relevance": round(data["relevance"] / c, 4)
            }

        return {
            "overall_averages": averages,
            "breakdown_by_query_type": type_breakdown,
            "breakdown_by_language": lang_breakdown
        }
