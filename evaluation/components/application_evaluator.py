"""
Application Component Evaluator
===============================
Evaluates user-facing application quality strictly on:
- Correctness (DeepEval GEval): Factual accuracy against reference answers.
- Completeness (DeepEval GEval): Breadth and thoroughness in answering the prompt.
- Toxicity (DeepEval ToxicityMetric): Civility, safety, and respectfulness.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

from evaluation.config import EvaluationConfig
from evaluation.client import RAGClient
from evaluation.components.generator_evaluator import GeneratorEvaluator
from evaluation.metrics.application_metrics import (
    compute_correctness,
    compute_completeness,
    compute_toxicity
)
from services.evaluation_service import GroqDeepEvalModel
from core.tracing import traceable


class ApplicationEvaluator:
    """Evaluator for Application-level response quality: Correctness, Completeness, Toxicity."""

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
        self.generator_eval = GeneratorEvaluator(
            router=self.router, client=self.client, config=self.config
        )

    @traceable(run_type="chain", name="eval_application_query")
    def evaluate_record(
        self,
        question: str,
        actual_output: str,
        expected_output: str,
        lang: str = "gu"
    ) -> Dict[str, Any]:
        """Evaluates a single question-answer pair on Correctness, Completeness, and Toxicity."""
        # 1. Correctness
        c_res = compute_correctness(
            question=question,
            actual_output=actual_output,
            expected_output=expected_output,
            model=self.judge_model,
            threshold=getattr(self.config, "correctness_threshold", 0.70)
        )

        # 2. Completeness
        comp_res = compute_completeness(
            question=question,
            actual_output=actual_output,
            expected_output=expected_output,
            model=self.judge_model,
            threshold=getattr(self.config, "completeness_threshold", 0.70)
        )

        # 3. Toxicity
        t_res = compute_toxicity(
            question=question,
            actual_output=actual_output,
            model=self.judge_model,
            threshold=getattr(self.config, "toxicity_threshold", 0.70)
        )

        w_c = 0.40
        w_comp = 0.35
        w_t = 0.25

        composite_score = round(
            (c_res["score"] * w_c) + (comp_res["score"] * w_comp) + (t_res["score"] * w_t),
            4
        )

        return {
            "query": question,
            "language": lang,
            "actual_output": actual_output,
            "expected_output": expected_output,
            "composite_score": composite_score,
            "metrics": {
                "correctness": c_res["score"],
                "correctness_reason": c_res.get("reason", ""),
                "completeness": comp_res["score"],
                "completeness_reason": comp_res.get("reason", ""),
                "toxicity": t_res["score"],
                "toxicity_reason": t_res.get("reason", "")
            }
        }

    def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Runs application evaluation across a dataset."""
        results = []
        iterator = tqdm(records, desc="Evaluating Application Quality") if show_progress else records

        by_type_acc = defaultdict(lambda: {"count": 0, "correctness": 0.0, "completeness": 0.0, "toxicity": 0.0, "composite": 0.0})
        by_lang_acc = defaultdict(lambda: {"count": 0, "correctness": 0.0, "completeness": 0.0, "toxicity": 0.0, "composite": 0.0})

        for record in iterator:
            question = record.get("question", "")
            lang = record.get("language", "gu")
            q_type = record.get("query_type", "UNKNOWN")
            expected_output = record.get("ground_truth_answer", record.get("expected_output", ""))

            # Get generated answer (from pre-generated answer or generate live)
            try:
                actual_output = record.get("generated_answer", "") if self.config.use_cached else ""
                if not actual_output:
                    # Live production pipeline (retrieve + generate), same as the API
                    actual_output = self.client.generate(question, language=lang, top_k=5).get("answer", "")

                item = self.evaluate_record(
                    question=question,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    lang=lang
                )
                item["query_id"] = record.get("query_id")
                item["query_type"] = q_type
                results.append(item)

                m = item["metrics"]
                for acc in (by_type_acc[q_type], by_lang_acc[lang]):
                    acc["count"] += 1
                    acc["correctness"] += m["correctness"]
                    acc["completeness"] += m["completeness"]
                    acc["toxicity"] += m["toxicity"]
                    acc["composite"] += item["composite_score"]

            except Exception as e:
                print(f"⚠️ Error evaluating application on '{question[:30]}...': {e}")

        n = len(results)
        if n == 0:
            return {"evaluated_queries": 0, "component": "application"}

        avg_comp = round(sum(r["composite_score"] for r in results) / n, 4)
        avg_correctness = round(sum(r["metrics"]["correctness"] for r in results) / n, 4)
        avg_completeness = round(sum(r["metrics"]["completeness"] for r in results) / n, 4)
        avg_toxicity = round(sum(r["metrics"]["toxicity"] for r in results) / n, 4)

        # Breakdowns
        by_type = {}
        for qt, d in by_type_acc.items():
            cnt = d["count"]
            by_type[qt] = {
                "count": cnt,
                "composite_score": round(d["composite"] / cnt, 4),
                "correctness": round(d["correctness"] / cnt, 4),
                "completeness": round(d["completeness"] / cnt, 4),
                "toxicity": round(d["toxicity"] / cnt, 4)
            }

        by_lang = {}
        for lg, d in by_lang_acc.items():
            cnt = d["count"]
            by_lang[lg] = {
                "count": cnt,
                "composite_score": round(d["composite"] / cnt, 4),
                "correctness": round(d["correctness"] / cnt, 4),
                "completeness": round(d["completeness"] / cnt, 4),
                "toxicity": round(d["toxicity"] / cnt, 4)
            }

        return {
            "component": "application",
            "evaluated_queries": n,
            "aggregates": {
                "overall_averages": {
                    "composite_score": avg_comp,
                    "correctness": avg_correctness,
                    "completeness": avg_completeness,
                    "toxicity": avg_toxicity
                },
                "breakdown_by_query_type": by_type,
                "breakdown_by_language": by_lang
            },
            "details": results
        }
