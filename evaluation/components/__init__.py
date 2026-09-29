"""
Evaluation Components Package
=============================
Contains isolated component evaluators:
- RetrieverEvaluator: Evaluates retrieval quality (Contextual Recall, Contextual Precision, Hit@K, MRR)
- GeneratorEvaluator: Evaluates generation quality (Faithfulness, Answer Relevancy, Semantic Similarity, F1)
"""

from .retriever_evaluator import RetrieverEvaluator
from .generator_evaluator import GeneratorEvaluator
