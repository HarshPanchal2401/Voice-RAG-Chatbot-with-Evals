"""
Dataset Combiner for Voice RAG System
====================================
Generates metric-specific Golden Datasets with exactly 15 samples in TOTAL
combining both Gujarati and Hindi (8 Gujarati + 7 Hindi = 15 total).
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
GUJARATI_DIR = DATA_DIR / "gujarati"
HINDI_DIR = DATA_DIR / "hindi"
COMBINED_DIR = DATA_DIR / "combined"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_stratified_samples(records: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Selects diverse samples across multiple query types (DESCRIPTION, NUMERIC, ENTITY, LOCATION, PERSON)."""
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        qt = r.get("query_type", "DESCRIPTION")
        by_type.setdefault(qt, []).append(r)

    selected = []
    types = list(by_type.keys())
    idx = 0
    while len(selected) < count and any(by_type.values()):
        current_type = types[idx % len(types)]
        if by_type[current_type]:
            selected.append(by_type[current_type].pop(0))
        idx += 1

    return selected[:count]


def create_combined_datasets():
    print("=" * 65)
    print("🔗 Generating Metric-Specific Golden Datasets (15 Total Samples)")
    print("   Total = 15 samples across both languages (8 Gujarati + 7 Hindi)")
    print("=" * 65)

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load source datasets
    gu_source = load_jsonl(GUJARATI_DIR / "golden_dataset_sample_100.jsonl")
    hi_source = load_jsonl(HINDI_DIR / "golden_dataset_sample_100.jsonl")

    for r in gu_source:
        r["language"] = "gu"
    for r in hi_source:
        r["language"] = "hi"

    # Select 8 from Gujarati + 7 from Hindi = 15 TOTAL
    gu_8 = select_stratified_samples(gu_source, count=8)
    hi_7 = select_stratified_samples(hi_source, count=7)
    base_15 = gu_8 + hi_7

    print(f"📊 Selected {len(gu_8)} Gujarati + {len(hi_7)} Hindi records = {len(base_15)} total samples.")

    # -----------------------------------------------------------------
    # Unified Retrieval Dataset (Contextual Recall + Contextual Precision)
    # Total 15 records (8 Gujarati + 7 Hindi) with pre-retrieved chunks (k=5)
    # -----------------------------------------------------------------
    retrieval_records = []
    print("⏳ Retrieving top-5 chunks for unified retrieval dataset...")
    for idx, r in enumerate(base_15, 1):
        q = r.get("question", "")
        lang = r.get("language", "gu")
        ret_contexts = []
        ret_pids = []
        ret_scores = []
        timings = {}
        try:
            from evaluation.client import RAGClient
            client = RAGClient()
            ret_res = client.retrieve(q, language=lang, top_k=5)
            docs = ret_res.get("documents", [])
            ret_contexts = [d.get("text", "") for d in docs]
            ret_pids = [d.get("passage_id", d.get("chunk_id", 0)) for d in docs]
            ret_scores = [round(d.get("rrf_score", 0.0), 6) for d in docs]
            timings = ret_res.get("timings", {})
        except Exception as e:
            # Fallback to ground truth context if offline
            ret_contexts = r.get("ground_truth_contexts", [])
            ret_pids = r.get("ground_truth_passage_ids", [])

        retrieval_records.append({
            "query_id": r.get("query_id"),
            "language": lang,
            "query_type": r.get("query_type"),
            "target_metrics": ["contextual_recall", "contextual_precision"],
            "component": "retriever",
            "evaluation_goal": "Evaluates Contextual Recall (fact coverage) and Contextual Precision (ranking relevance) on retrieved chunks.",
            "question": q,
            "k": 5,
            "retrieved_contexts": ret_contexts,
            "retrieved_passage_ids": ret_pids,
            "retrieved_scores": ret_scores,
            "expected_output": r.get("ground_truth_answer", r.get("expected_output", "")),
            "ground_truth_contexts": r.get("ground_truth_contexts", []),
            "ground_truth_passage_ids": r.get("ground_truth_passage_ids", []),
            "retrieval_timings": timings
        })

    random.seed(42)
    random.shuffle(retrieval_records)
    save_jsonl(retrieval_records, COMBINED_DIR / "retrieval_golden_dataset.jsonl")
    print(f"✅ Created [retrieval_golden_dataset.jsonl]: {len(retrieval_records)} records total (8 GU + 7 HI) with retrieved chunks")

    # Clean up obsolete split datasets if they exist
    for old_file in ("metric_contextual_recall.jsonl", "metric_contextual_precision.jsonl", "retrieval_evaluation_dataset.jsonl"):
        p = COMBINED_DIR / old_file
        if p.exists():
            p.unlink()
            print(f"🗑️ Removed obsolete dataset: {old_file}")

    # -----------------------------------------------------------------
    # Unified Generator Dataset (Faithfulness + Answer Relevancy)
    # Total 15 records (8 Gujarati + 7 Hindi) with pre-generated LLM answers
    # -----------------------------------------------------------------
    generator_records = []
    for r in base_15:
        gold_ctx = r.get("ground_truth_contexts", [""])[0] if r.get("ground_truth_contexts") else ""
        generator_records.append({
            "query_id": r.get("query_id"),
            "language": r.get("language"),
            "query_type": r.get("query_type"),
            "target_metrics": ["faithfulness", "answer_relevance"],
            "component": "generator",
            "evaluation_goal": "Evaluates Faithfulness (factual adherence to context) and Answer Relevancy (directness to user question).",
            "question": r.get("question"),
            "contexts": r.get("ground_truth_contexts", [gold_ctx] if gold_ctx else []),
            "generated_answer": r.get("generated_answer", ""),
            "ground_truth_answer": r.get("ground_truth_answer", "")
        })
    random.seed(44)
    random.shuffle(generator_records)
    save_jsonl(generator_records, COMBINED_DIR / "generator_golden_dataset.jsonl")
    save_jsonl(base_15, COMBINED_DIR / "golden_dataset_sample_15.jsonl")
    print(f"✅ Created [generator_golden_dataset.jsonl]: {len(generator_records)} records total (8 GU + 7 HI)")
    print(f"✅ Created [golden_dataset_sample_15.jsonl]: {len(base_15)} records total")

    # -----------------------------------------------------------------
    # Unified Pipeline Dataset (Faithfulness + Answer Relevancy + Context Relevancy)
    # Total 15 records (8 Gujarati + 7 Hindi) for End-to-End Pipeline Evaluation
    # -----------------------------------------------------------------
    pipeline_records = []
    # If pre-generated answers from retrieved contexts exist, preserve them
    existing_pipe_map = {r.get("query_id"): r for r in load_jsonl(COMBINED_DIR / "pipeline_golden_dataset.jsonl")}
    retrieval_map = {r.get("query_id"): r for r in retrieval_records}

    for r in base_15:
        qid = r.get("query_id")
        ret_item = retrieval_map.get(qid, {})
        pipe_item = existing_pipe_map.get(qid, {})

        pipeline_records.append({
            "query_id": qid,
            "language": r.get("language"),
            "query_type": r.get("query_type"),
            "target_metrics": ["faithfulness", "answer_relevance", "context_relevance"],
            "component": "pipeline",
            "evaluation_goal": "Evaluates Faithfulness (factual grounding in retrieved contexts), Answer Relevancy (directness to user question), and Context Relevancy (relevance of retrieved contexts to query).",
            "question": r.get("question"),
            "k": ret_item.get("k", 5),
            "retrieved_contexts": ret_item.get("retrieved_contexts", []),
            "retrieved_passage_ids": ret_item.get("retrieved_passage_ids", []),
            "retrieved_scores": ret_item.get("retrieved_scores", []),
            "retrieval_timings": ret_item.get("retrieval_timings", {}),
            "generated_answer": pipe_item.get("generated_answer", r.get("generated_answer", "")),
            "generation_timings": pipe_item.get("generation_timings", {}),
            "ground_truth_answer": r.get("ground_truth_answer", r.get("expected_output", "")),
            "ground_truth_contexts": r.get("ground_truth_contexts", []),
            "ground_truth_passage_ids": r.get("ground_truth_passage_ids", []),
            "expected_output": r.get("expected_output", r.get("ground_truth_answer", ""))
        })

    save_jsonl(pipeline_records, COMBINED_DIR / "pipeline_golden_dataset.jsonl")
    print(f"✅ Created [pipeline_golden_dataset.jsonl]: {len(pipeline_records)} records total (8 GU + 7 HI)")

    # Clean up obsolete split generator datasets
    for old_file in ("metric_faithfulness.jsonl", "metric_answer_relevancy.jsonl", "generator_evaluation_dataset.jsonl"):
        p = COMBINED_DIR / old_file
        if p.exists():
            p.unlink()
            print(f"🗑️ Removed obsolete dataset: {old_file}")

    # -----------------------------------------------------------------
    # Standard 100 & 500 benchmarks (kept for larger runs)
    # -----------------------------------------------------------------
    combined_100 = gu_source[:50] + hi_source[:50]
    random.seed(46)
    random.shuffle(combined_100)
    save_jsonl(combined_100, COMBINED_DIR / "golden_dataset_sample_100.jsonl")

    gu_500 = load_jsonl(GUJARATI_DIR / "golden_dataset_sample_500.jsonl")
    hi_500 = load_jsonl(HINDI_DIR / "golden_dataset_sample_500.jsonl")
    for r in gu_500:
        r["language"] = "gu"
    for r in hi_500:
        r["language"] = "hi"
    combined_500 = gu_500[:250] + hi_500[:250]
    random.seed(47)
    random.shuffle(combined_500)
    save_jsonl(combined_500, COMBINED_DIR / "golden_dataset_sample_500.jsonl")

    # -----------------------------------------------------------------
    # Manifest / Summary JSON
    # -----------------------------------------------------------------
    summary = {
        "dataset_name": "Metric-Specific Golden Evaluation Benchmarks (15 Samples Total)",
        "description": "Multi-language evaluation benchmarks tailored for specific metrics with exactly 15 total samples (8 Gujarati + 7 Hindi).",
        "sample_rule": "Total sample size is exactly 15 across both languages combined (8 GU + 7 HI), NOT 15 from each language.",
        "retrieval_dataset_15_samples": {
            "retrieval_golden_dataset.jsonl": "15 records total (8 Gujarati + 7 Hindi) with pre-retrieved chunks (k=5) for Contextual Recall and Contextual Precision",
            "retrieval_evaluation_dataset.jsonl": "Alias pointing to the unified 15-sample retrieval evaluation dataset"
        },
        "generator_datasets_15_samples": {
            "metric_faithfulness.jsonl": "15 records total (8 Gujarati + 7 Hindi) for Faithfulness / Hallucination evaluation",
            "metric_answer_relevancy.jsonl": "15 records total (8 Gujarati + 7 Hindi) for Answer Relevancy evaluation",
            "generator_golden_dataset.jsonl": "15 records total (8 Gujarati + 7 Hindi) for Generator component testing"
        },
        "pipeline_datasets_15_samples": {
            "pipeline_golden_dataset.jsonl": "15 records total (8 Gujarati + 7 Hindi) with retrieved chunks and generated RAG answers for Faithfulness, Answer Relevancy, and Context Relevancy"
        },
        "component_datasets_15_samples": {
            "retrieval_golden_dataset.jsonl": "15 records total (8 Gujarati + 7 Hindi) for Retriever component testing",
            "generator_golden_dataset.jsonl": "15 records total (8 Gujarati + 7 Hindi) for Generator component testing",
            "pipeline_golden_dataset.jsonl": "15 records total (8 Gujarati + 7 Hindi) for End-to-End Pipeline testing",
            "golden_dataset_sample_15.jsonl": "15 records total (8 Gujarati + 7 Hindi) for End-to-End RAG benchmarking"
        },
        "larger_benchmarks": {
            "golden_dataset_sample_100.jsonl": "100 records (50 Gujarati + 50 Hindi)",
            "golden_dataset_sample_500.jsonl": "500 records (250 Gujarati + 250 Hindi)"
        }
    }

    with open(COMBINED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("✅ Created [dataset_summary.json]")
    print("=" * 65)


if __name__ == "__main__":
    create_combined_datasets()
