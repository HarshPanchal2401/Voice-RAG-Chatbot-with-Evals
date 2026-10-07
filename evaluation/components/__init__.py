"""
Evaluation Components Package
=============================
- RetrieverEvaluator  : doc-key IR metrics (hit/recall/precision/MRR/nDCG) + contextual judge metrics,
                        optional paired rerank-vs-no-rerank comparison
- GeneratorEvaluator  : faithfulness / answer relevance / token F1 / refusal metrics (oracle or retrieved)
- ApplicationEvaluator: correctness / completeness (+ toxicity gate)
"""

from .retriever_evaluator import RetrieverEvaluator
from .generator_evaluator import GeneratorEvaluator
from .application_evaluator import ApplicationEvaluator

__all__ = ["RetrieverEvaluator", "GeneratorEvaluator", "ApplicationEvaluator"]
