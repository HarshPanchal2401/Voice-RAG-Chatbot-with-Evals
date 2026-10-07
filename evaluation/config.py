"""
Evaluation Configuration & Defaults
===================================
* Model roles come from ``core.config`` (GROQ_MODEL_NAME / RERANKER_MODEL_NAME /
  DEFAULT_JUDGE_MODEL / RERANKER_BACKEND) with env fallbacks - see
  ``evaluation.metrics.judge.resolve_model_roles``. The judge must differ from the
  generator and the LLM reranker.
* Thresholds used for PASS / BELOW / INCONCLUSIVE status in reports live here and can be
  overridden with env vars ``RAG_EVAL_THRESHOLD_<METRIC>`` (e.g. RAG_EVAL_THRESHOLD_FAITHFULNESS=0.8)
  and ``RAG_EVAL_P95_<NAME>_MS`` for latency budgets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.metrics.judge import ModelRoles, resolve_model_roles

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Quality thresholds (minimum acceptable mean). Status: PASS if the 95% CI lower bound is
# >= threshold, BELOW if the mean is < threshold, INCONCLUSIVE otherwise.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "hit_rate": 0.80,
    "recall_at_k": 0.70,
    "mrr": 0.60,
    "ndcg": 0.60,
    "contextual_recall": 0.70,
    "contextual_precision": 0.70,
    "context_relevance": 0.50,
    "faithfulness": 0.80,
    "answer_relevance": 0.70,
    "answer_correctness": 0.70,
    "completeness": 0.70,
    "composite_score": 0.70,
    "toxicity_safety": 0.99,
}

# Metrics where lower is better (status logic is inverted).
LOWER_IS_BETTER = {"refusal_rate": 0.10, "refused_with_gold_retrieved": 0.05, "lexical_toxicity_flag": 0.0}

# p95 latency budgets (ms) excluding warm-up.
DEFAULT_LATENCY_P95_MS: Dict[str, float] = {
    "retrieval_total_ms": 800.0,
    "ttft_ms": 1500.0,
    "llm_ms": 3000.0,
    "total_ms": 4000.0,
}


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_thresholds() -> Dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    for k in list(out):
        v = _env_float(f"RAG_EVAL_THRESHOLD_{k.upper()}")
        if v is not None:
            out[k] = v
    return out


def load_lower_is_better() -> Dict[str, float]:
    out = dict(LOWER_IS_BETTER)
    for k in list(out):
        v = _env_float(f"RAG_EVAL_THRESHOLD_{k.upper()}")
        if v is not None:
            out[k] = v
    return out


def load_latency_budgets() -> Dict[str, float]:
    out = dict(DEFAULT_LATENCY_P95_MS)
    for k in list(out):
        v = _env_float(f"RAG_EVAL_P95_{k.upper()}")
        if v is not None:
            out[k] = v
    return out


@dataclass
class EvaluationConfig:
    language: str = "combined"            # gu | hi | combined
    mode: str = "server"                  # server | inprocess (no silent fallback)
    base_url: str = "http://localhost:8000"
    use_cached: bool = False              # score a saved snapshot instead of the live pipeline
    allow_unknown_generator: bool = False # allow cached answers with unknown generator model
    default_k: int = 5
    seed: int = 42
    concurrency: int = 4                  # concurrent judge-scored records
    warmup: int = 1                       # requests excluded from latency percentiles
    max_error_rate: float = 0.05          # run fails above this pipeline-error rate
    judge_cache: bool = True
    n_boot: int = 2000
    judge_model_override: Optional[str] = None
    groq_api_key: Optional[str] = None
    roles: Optional[ModelRoles] = None
    thresholds: Dict[str, float] = field(default_factory=load_thresholds)
    lower_is_better: Dict[str, float] = field(default_factory=load_lower_is_better)
    latency_budgets: Dict[str, float] = field(default_factory=load_latency_budgets)
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR

    def __post_init__(self):
        if not self.groq_api_key:
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if self.roles is None:
            self.roles = resolve_model_roles(judge_override=self.judge_model_override)
        if self.mode not in ("server", "inprocess"):
            raise ValueError(f"mode must be 'server' or 'inprocess', got {self.mode!r}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def judge_model_name(self) -> str:
        return self.roles.judge

    @property
    def generator_model_name(self) -> str:
        return self.roles.generator


def __getattr__(name: str) -> Any:
    """Backwards compatibility: GROQ_MODEL_NAME / SYSTEM_PROMPTS / JUDGE_MODEL_NAME."""
    if name in ("GROQ_MODEL_NAME", "SYSTEM_PROMPTS"):
        import pipeline.generator as gen

        return getattr(gen, name)
    if name == "JUDGE_MODEL_NAME":
        return resolve_model_roles().judge
    raise AttributeError(name)
