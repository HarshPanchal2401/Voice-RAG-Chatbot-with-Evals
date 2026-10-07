"""
Evaluation Metrics Package (single source of truth for offline AND online evaluation)
====================================================================================
Modules:
    text_utils          Unicode-aware tokenizer, refusal + toxicity lexicons
    judge               Groq judge (retries, async, rate limit, cache), model roles
    base                metric outcome + the one failure policy
    retrieval_metrics   doc_key relevance, hit/recall/precision/MRR/nDCG, contextual metrics
    generation_metrics  faithfulness, answer relevance, token F1, refusal metrics
    application_metrics correctness / completeness (GEval with explicit steps), toxicity gate
    composite           the one documented composite score
    stats               bootstrap CIs, paired bootstrap, sign test, latency percentiles
    aggregate           per-metric aggregation with CIs and judge success rates

Submodules are imported lazily so that the lightweight helpers (tokenizer, stats)
do not pull in DeepEval.
"""

import importlib

_EXPORTS = {
    # retrieval
    "make_doc_key": "retrieval_metrics",
    "gold_doc_keys": "retrieval_metrics",
    "doc_keys_from_documents": "retrieval_metrics",
    "compute_hit_rate": "retrieval_metrics",
    "compute_recall_at_k": "retrieval_metrics",
    "compute_precision_at_k": "retrieval_metrics",
    "compute_mrr": "retrieval_metrics",
    "compute_ndcg": "retrieval_metrics",
    "compute_context_jaccard": "retrieval_metrics",
    "compute_contextual_recall": "retrieval_metrics",
    "compute_contextual_precision": "retrieval_metrics",
    "compute_context_relevance": "retrieval_metrics",
    "evaluate_retrieval_record": "retrieval_metrics",
    # generation
    "compute_faithfulness": "generation_metrics",
    "compute_answer_relevance": "generation_metrics",
    "compute_indic_token_f1": "generation_metrics",
    "compute_semantic_similarity": "generation_metrics",
    "evaluate_generation_record": "generation_metrics",
    # application
    "compute_correctness": "application_metrics",
    "compute_completeness": "application_metrics",
    "compute_toxicity": "application_metrics",
    # text
    "tokenize": "text_utils",
    "is_refusal": "text_utils",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)
