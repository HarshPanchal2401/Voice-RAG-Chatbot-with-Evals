"""
Evaluation Metrics Package
==========================
Component-based metrics for Retrieval and Generation evaluation.
"""

from .retrieval_metrics import (
    compute_contextual_recall,
    compute_contextual_precision,
    compute_hit_rate,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    compute_context_jaccard,
    evaluate_retrieval_record,
)

from .generation_metrics import (
    compute_faithfulness,
    compute_answer_relevance,
    compute_indic_token_f1,
    compute_semantic_similarity,
    evaluate_generation_record,
)
