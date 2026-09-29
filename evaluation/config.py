"""
Evaluation Configuration & Defaults
===================================
Defines settings, metric thresholds, and model parameters for custom evaluation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


# Prompts and answering model come from the production generation layer, so the
# evaluation suite always tests exactly what the API serves.
from pipeline.generator import GROQ_MODEL_NAME, SYSTEM_PROMPTS  # noqa: E402,F401

JUDGE_MODEL_NAME = os.environ.get("RAG_JUDGE_MODEL", "openai/gpt-oss-120b")


@dataclass
class EvaluationConfig:
    # Languages
    language: str = "combined"  # "gu", "hi", or "combined"
    supported_languages: List[str] = field(default_factory=lambda: ["gu", "hi", "combined"])

    # LLM Judge settings
    judge_model_name: str = JUDGE_MODEL_NAME
    groq_api_key: Optional[str] = None
    temperature: float = 0.0

    # Metric thresholds
    contextual_recall_threshold: float = 0.7
    contextual_precision_threshold: float = 0.7
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    context_relevance_threshold: float = 0.5
    correctness_threshold: float = 0.7
    completeness_threshold: float = 0.7
    toxicity_threshold: float = 0.7

    # False (default): run the LIVE production pipeline for every record, so scores and
    # latency reflect the current code. True: reuse retrieved_contexts / generated_answer /
    # timings stored in the golden dataset files (fast, but a frozen snapshot).
    use_cached: bool = False

    # Retrieval settings
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    default_k: int = 5

    # Evaluation paths
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR

    # Composite weights for pipeline score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.35,
        "answer_relevance": 0.35,
        "context_relevance": 0.30,
        "contextual_recall": 0.25,
        "contextual_precision": 0.25
    })

    def __post_init__(self):
        if not self.groq_api_key:
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
