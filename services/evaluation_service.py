"""
DeepEval Evaluation Engine Service
==================================
Production-grade LLM evaluation engine powered by DeepEval.
Evaluates Indic & English RAG systems using:
- 🛡️ FaithfulnessMetric (Hallucinations & Context Groundedness)
- 🎯 AnswerRelevancyMetric (Question alignment & redundancy reduction)
- 📚 ContextualRecallMetric (Ground truth fact retrieval coverage)
- ⚖️ GEval (Fact-checked Answer Correctness)
- 🌐 BGE-M3 Semantic Cosine Similarity
"""

import os
import sys
import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from core.config import DEFAULT_JUDGE_MODEL
from core.tracing import get_llm_client, traceable

load_dotenv()

# UTF-8 encoding support for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# DeepEval imports
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as SingleTurnParams

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    GEval
)


class GroqDeepEvalModel(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM backend leveraging Groq's ultra-fast, OpenAI-compatible API.
    Provides deterministic temperature=0.0 evaluation reasoning for Indic and English RAG.
    """

    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        super().__init__(model_name)

    def load_model(self):
        return get_llm_client()

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        messages = []
        is_json_prompt = (schema is not None) or ("json" in prompt.lower()) or ("verdict" in prompt.lower())
        if is_json_prompt:
            messages.append({
                "role": "system",
                "content": "You are a specialized evaluation judge. Output ONLY valid JSON adhering strictly to the requested schema."
            })
        messages.append({"role": "user", "content": prompt})

        call_params = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        if is_json_prompt:
            call_params["response_format"] = {"type": "json_object"}

        candidate_models = [self.model_name, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound-mini"]
        candidate_models = list(dict.fromkeys(candidate_models))

        last_err = ""
        for m in candidate_models:
            call_params["model"] = m
            try:
                resp = self.model.chat.completions.create(**call_params)
                content = (resp.choices[0].message.content or "").strip()

                fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if fence_match:
                    content = fence_match.group(1).strip()
                elif content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n?", "", content)
                    content = re.sub(r"\n?```$", "", content).strip()

                if is_json_prompt and "{" in content and "}" in content:
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    content = content[start_idx:end_idx].strip()

                return content
            except Exception as e:
                last_err = str(e)
                continue
        return f'{{"error": "{last_err}"}}'

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        return self.generate(prompt, schema=schema, *args, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


def normalize_text(text: str, lang: str = "gu") -> str:
    """Standardizes punctuation and whitespace for fuzzy matching."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\?\.!,\-_:;।\(\)\[\]\{\}\"\'`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DeepEvalEvaluator:
    """
    DeepEval-powered evaluation engine for Multi-Language Voice & Text RAG.
    Maintains Golden Dataset matching and runs DeepEval metrics with structured reasoning.
    """

    def __init__(
        self,
        golden_dataset_path: Optional[str] = None,
        groq_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        eval_llm_model: Optional[str] = None,
        language: str = "gu"
    ):
        self.language = language.lower().strip()
        self.embedding_model = embedding_model
        self.eval_llm_model = eval_llm_model if eval_llm_model and eval_llm_model != os.environ.get("RAG_LLM_MODEL", "qwen/qwen3.8-27b") else DEFAULT_JUDGE_MODEL

        api_key = os.environ.get("GROQ_API_KEY")
        self.deepeval_model = GroqDeepEvalModel(
            model_name=self.eval_llm_model,
            api_key=api_key
        )

        self.golden_records: List[Dict[str, Any]] = []
        self.golden_by_normalized_q: Dict[str, Dict[str, Any]] = {}
        self.golden_by_qid: Dict[Any, Dict[str, Any]] = {}

        if golden_dataset_path:
            self.load_golden_dataset(golden_dataset_path)

    def load_golden_dataset(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ Warning: Golden dataset file not found at {path}")
            return

        count = 0
        self.golden_records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = record.get("query_id")
                q_text = record.get("question", "")
                norm_q = normalize_text(q_text, self.language)

                if norm_q:
                    self.golden_by_normalized_q[norm_q] = record
                if qid is not None:
                    self.golden_by_qid[qid] = record
                self.golden_records.append(record)
                count += 1

        print(f"📊 [DeepEvalEvaluator ({self.language.upper()})] Loaded {count:,} golden evaluation records from {path.name}")

    def get_random_samples(self, count: int = 20) -> list:
        import random
        if not self.golden_records:
            return []
        k = min(count, len(self.golden_records))
        samples = random.sample(self.golden_records, k)
        return [
            {
                "query_id": r.get("query_id"),
                "question": r.get("question"),
                "query_type": r.get("query_type"),
                "ground_truth_answer": r.get("ground_truth_answer")
            }
            for r in samples
        ]

    def find_golden_record(self, query: str, query_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if query_id is not None and query_id in self.golden_by_qid:
            return self.golden_by_qid[query_id]

        norm_q = normalize_text(query, self.language)
        if norm_q in self.golden_by_normalized_q:
            return self.golden_by_normalized_q[norm_q]

        for g_norm, record in self.golden_by_normalized_q.items():
            if norm_q in g_norm or g_norm in norm_q:
                overlap_ratio = min(len(norm_q), len(g_norm)) / max(len(norm_q), len(g_norm))
                if overlap_ratio >= 0.75:
                    return record
        return None

    def compute_embedding_similarity(self, text_a: str, text_b: str) -> float:
        if not self.embedding_model or not text_a or not text_b:
            return 0.0
        try:
            emb_a = self.embedding_model.encode_query(text_a)[0]
            emb_b = self.embedding_model.encode_query(text_b)[0]
            sim = float(np.dot(emb_a, emb_b))
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception:
            return 0.0

    def evaluate_open_query(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: List[str]
    ) -> Dict[str, Any]:
        contexts = retrieved_contexts if retrieved_contexts else ["No context retrieved."]
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            retrieval_context=contexts
        )

        f_metric = FaithfulnessMetric(threshold=0.7, model=self.deepeval_model, include_reason=True)
        try:
            f_metric.measure(test_case)
            f_score = round(float(f_metric.score), 4)
            f_reason = f_metric.reason or ""
        except Exception as e:
            f_score = 0.0
            f_reason = f"Faithfulness evaluation error: {str(e)}"

        r_metric = AnswerRelevancyMetric(threshold=0.7, model=self.deepeval_model, include_reason=True)
        try:
            r_metric.measure(test_case)
            r_score = round(float(r_metric.score), 4)
            r_reason = r_metric.reason or ""
        except Exception as e:
            r_score = 0.0
            r_reason = f"Relevancy evaluation error: {str(e)}"

        overall = round((f_score * 0.5) + (r_score * 0.5), 4)
        combined_reason = f_reason if f_reason else r_reason

        return {
            "faithfulness": f_score,
            "answer_relevance": r_score,
            "overall_score": overall,
            "explanation": combined_reason,
            "reason": combined_reason
        }

    def evaluate_golden_query(
        self,
        question: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_contexts: List[str],
        retrieved_passage_ids: Optional[List[int]] = None,
        gt_passage_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        contexts = retrieved_contexts if retrieved_contexts else ["No context retrieved."]
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            expected_output=ground_truth_answer,
            retrieval_context=contexts
        )

        reasons = []

        f_metric = FaithfulnessMetric(threshold=0.7, model=self.deepeval_model, include_reason=True)
        try:
            f_metric.measure(test_case)
            f_score = round(float(f_metric.score), 4)
            if f_metric.reason:
                reasons.append(f_metric.reason)
        except Exception as e:
            f_score = 0.0
            reasons.append(f"Faithfulness note: {str(e)}")

        r_metric = AnswerRelevancyMetric(threshold=0.7, model=self.deepeval_model, include_reason=True)
        try:
            r_metric.measure(test_case)
            r_score = round(float(r_metric.score), 4)
            if r_metric.reason:
                reasons.append(r_metric.reason)
        except Exception as e:
            r_score = 0.0
            reasons.append(f"Relevancy note: {str(e)}")

        cr_metric = ContextualRecallMetric(threshold=0.7, model=self.deepeval_model, include_reason=True)
        try:
            cr_metric.measure(test_case)
            cr_score = round(float(cr_metric.score), 4)
        except Exception as e:
            cr_score = 0.0
            reasons.append(f"Contextual recall error: {str(e)}")

        correctness_metric = GEval(
            name="Answer Correctness",
            criteria="Determine whether the actual output is factually accurate and consistent with the expected ground-truth output.",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            model=self.deepeval_model
        )
        try:
            correctness_metric.measure(test_case)
            c_score = round(float(correctness_metric.score), 4)
            if correctness_metric.reason:
                reasons.append(correctness_metric.reason)
        except Exception as e:
            c_score = 0.0
            reasons.append(f"Correctness note: {str(e)}")

        emb_sim = self.compute_embedding_similarity(generated_answer, ground_truth_answer)

        hit_rate = 0.0
        if retrieved_passage_ids and gt_passage_ids:
            hits = set(retrieved_passage_ids).intersection(set(gt_passage_ids))
            hit_rate = 1.0 if len(hits) > 0 else 0.0

        overall = round(
            (c_score * 0.35) +
            (f_score * 0.25) +
            (r_score * 0.20) +
            (cr_score * 0.20),
            4
        )

        full_explanation = " | ".join(r for r in reasons if r)

        return {
            "answer_correctness": c_score,
            "answer_similarity": round(emb_sim, 4),
            "faithfulness": f_score,
            "answer_relevance": r_score,
            "context_recall": cr_score,
            "hit_rate_at_k": hit_rate,
            "overall_score": overall,
            "explanation": full_explanation,
            "reason": full_explanation
        }

    @traceable(run_type="chain", name="deepeval_metrics")
    def evaluate(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        retrieved_passage_ids: Optional[List[int]] = None,
        query_id: Optional[int] = None,
        allow_open_eval: bool = True
    ) -> Dict[str, Any]:
        golden_record = self.find_golden_record(question, query_id)

        if not golden_record:
            if allow_open_eval:
                open_metrics = self.evaluate_open_query(
                    question=question,
                    generated_answer=generated_answer,
                    retrieved_contexts=retrieved_contexts
                )
                return {
                    "is_golden": False,
                    "warning": "Query not in Golden Dataset. Evaluated with DeepEval for Faithfulness and Answer Relevancy.",
                    "scores": {
                        "faithfulness": open_metrics["faithfulness"],
                        "answer_relevance": open_metrics["answer_relevance"],
                        "overall_score": open_metrics["overall_score"],
                        "explanation": open_metrics.get("explanation", ""),
                        "reason": open_metrics.get("reason", "")
                    },
                    "ground_truth": None
                }
            return {
                "is_golden": False,
                "warning": "⚠️ Warning: Query not found in Golden Dataset! Evaluation metrics cannot be compared against ground truth.",
                "scores": None,
                "ground_truth": None
            }

        ground_truth_answer = golden_record.get("ground_truth_answer", "")
        ground_truth_contexts = golden_record.get("ground_truth_contexts", [])
        gt_passage_ids = golden_record.get("ground_truth_passage_ids", [])

        scores = self.evaluate_golden_query(
            question=question,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth_answer,
            retrieved_contexts=retrieved_contexts,
            retrieved_passage_ids=retrieved_passage_ids,
            gt_passage_ids=gt_passage_ids
        )

        return {
            "is_golden": True,
            "query_id": golden_record.get("query_id"),
            "query_type": golden_record.get("query_type"),
            "ground_truth_answer": ground_truth_answer,
            "ground_truth_contexts": ground_truth_contexts,
            "scores": scores
        }


# Aliases
RagasEvaluator = DeepEvalEvaluator
