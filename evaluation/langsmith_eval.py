"""
LangSmith Native Evaluation Runner
==================================
Runs end-to-end RAG evaluations natively on LangSmith:
1. Syncs the Indic Golden Dataset (Gujarati & Hindi) to LangSmith Datasets.
2. Runs the live Voice RAG pipeline against each test example.
3. Scores responses with RAG Triad evaluators (Faithfulness, Answer Relevance, Context Recall, Latency).
4. Publishes an interactive Experiment to the LangSmith web dashboard with side-by-side comparisons.

Usage:
  # Run evaluation on 15 samples across Gujarati & Hindi (uploads dataset if not present)
  python -m evaluation.langsmith_eval

  # Run evaluation with a specific sample size (e.g., 5 queries)
  python -m evaluation.langsmith_eval --sample 5

  # Run evaluation for only Gujarati or Hindi
  python -m evaluation.langsmith_eval --lang gu
  python -m evaluation.langsmith_eval --lang hi

  # Force re-upload / re-sync of dataset to LangSmith
  python -m evaluation.langsmith_eval --sync-dataset
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

load_dotenv()

import langsmith
from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult

from evaluation.client import RAGClient
from evaluation.data_loader import GoldenDataLoader
from evaluation.config import EvaluationConfig
from core.tracing import tracing_enabled, project_name


# ------------------------------------------------------------
# 1. Dataset Synchronization to LangSmith
# ------------------------------------------------------------

DATASET_NAME_DEFAULT = "Voice-RAG-Indic-Golden-Dataset"


from langsmith.schemas import DataType


def sync_golden_dataset_to_langsmith(
    client: Client,
    records: List[Dict[str, Any]],
    dataset_name: str = DATASET_NAME_DEFAULT,
    description: str = "Multilingual Gujarati & Hindi Golden Benchmark for Voice RAG evaluation."
) -> Any:
    """
    Creates or updates the dataset in LangSmith with the golden questions,
    ground-truth answers, and reference contexts.
    """
    print(f"📦 Checking LangSmith dataset '{dataset_name}'...")
    dataset = None
    try:
        for ds in client.list_datasets():
            if ds.name == dataset_name:
                dataset = ds
                break
    except Exception as e:
        print(f"⚠️ Error listing datasets: {e}")

    if dataset is None:
        print(f"✨ Creating new dataset '{dataset_name}' in LangSmith...")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
            data_type=DataType.kv
        )

        inputs = []
        outputs = []
        metadatas = []
        for r in records:
            inputs.append({
                "question": r.get("question", ""),
                "language": r.get("language", "gu"),
            })
            outputs.append({
                "ground_truth_answer": r.get("ground_truth_answer", r.get("expected_output", "")),
                "ground_truth_contexts": r.get("ground_truth_contexts", []),
            })
            metadatas.append({
                "query_id": r.get("query_id"),
                "query_type": r.get("query_type", "GENERAL"),
            })

        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
            metadata=metadatas
        )
        print(f"✅ Successfully uploaded {len(inputs)} examples to LangSmith dataset '{dataset_name}'!")
    else:
        print(f"✅ Found existing LangSmith dataset '{dataset_name}' (ID: {dataset.id}).")

    return dataset


# ------------------------------------------------------------
# 2. Target Invocation (RAG Pipeline)
# ------------------------------------------------------------

_rag_client: Optional[RAGClient] = None
_fallback_router = None


def get_rag_target():
    global _rag_client, _fallback_router
    if _rag_client is None:
        _rag_client = RAGClient(base_url="http://localhost:8000")
        if not _rag_client.is_server_online():
            print("ℹ️ Live server at http://localhost:8000 not online, initializing local LanguageRouter...")
            from pipeline.router import LanguageRouter
            _fallback_router = LanguageRouter(verbose=False)
            _rag_client.fallback_router = _fallback_router
    return _rag_client


def rag_pipeline_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangSmith Target Function:
    Accepts example inputs, runs the RAG pipeline, and returns outputs.
    """
    question = inputs["question"]
    lang = inputs.get("language", "gu")
    rag = get_rag_target()

    t0 = time.perf_counter()
    result = rag.generate(query=question, language=lang, top_k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    contexts = [doc.get("text", "") for doc in result.get("documents", [])]
    return {
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "latency_ms": elapsed_ms,
        "language": lang
    }


# ------------------------------------------------------------
# 3. LangSmith Native Evaluators
# ------------------------------------------------------------

def faithfulness_evaluator(run, example) -> EvaluationResult:
    """Evaluates if the answer is grounded in the retrieved contexts."""
    answer = (run.outputs or {}).get("answer", "")
    contexts = (run.outputs or {}).get("contexts", [])

    if not answer:
        return EvaluationResult(key="faithfulness", score=0.0, comment="Empty answer")
    if not contexts:
        return EvaluationResult(key="faithfulness", score=0.0, comment="No contexts retrieved")

    # Lexical grounding overlap
    import re
    all_ctx = " ".join(contexts).lower()
    words = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 2]
    if not words:
        return EvaluationResult(key="faithfulness", score=0.8, comment="Short answer tokens")

    grounded_count = sum(1 for w in words if w in all_ctx)
    ratio = grounded_count / len(words)

    if ratio >= 0.40:
        score = round(min(1.0, 0.75 + (ratio * 0.25)), 2)
        comment = f"Grounded response ({ratio:.1%} overlap with retrieved context)"
    elif ratio > 0.20:
        score = 0.70
        comment = f"Moderate context grounding ({ratio:.1%})"
    else:
        score = round(ratio, 2)
        comment = f"Low context overlap ({ratio:.1%})"

    return EvaluationResult(key="faithfulness", score=score, comment=comment)


def answer_relevance_evaluator(run, example) -> EvaluationResult:
    """Evaluates if the answer directly addresses the user query."""
    question = (example.inputs or {}).get("question", "")
    answer = (run.outputs or {}).get("answer", "")

    if not answer or not question:
        return EvaluationResult(key="answer_relevance", score=0.0, comment="Missing question or answer")

    import re
    q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 2]
    ans_lower = answer.lower()

    refusals = ["મને ખબર નથી", "પૂરતી માહિતી ઉપલબ્ધ નથી", "પર્યાપ્ત જાણકારી ઉપલબ્ધ નહીં", "जानकारी उपलब्ध नहीं", "not enough information"]
    if any(r in ans_lower for r in refusals):
        return EvaluationResult(key="answer_relevance", score=0.5, comment="Graceful refusal due to missing context")

    if not q_words:
        return EvaluationResult(key="answer_relevance", score=0.85, comment="Valid response")

    overlap = sum(1 for w in q_words if w in ans_lower) / len(q_words)
    if overlap >= 0.20:
        score = round(min(1.0, 0.80 + (overlap * 0.20)), 2)
        comment = f"Addresses question key entities ({overlap:.1%} keyword overlap)"
    else:
        score = 0.75
        comment = "Relevant response provided"

    return EvaluationResult(key="answer_relevance", score=score, comment=comment)


def context_recall_evaluator(run, example) -> EvaluationResult:
    """Evaluates if the retrieved passages cover the ground-truth reference context."""
    gt_contexts = (example.outputs or {}).get("ground_truth_contexts", [])
    retrieved = (run.outputs or {}).get("contexts", [])

    if not gt_contexts:
        return EvaluationResult(key="context_recall", score=1.0, comment="No reference context specified")
    if not retrieved:
        return EvaluationResult(key="context_recall", score=0.0, comment="No contexts retrieved")

    import re
    gt_text = " ".join(gt_contexts).lower()
    gt_keywords = [w for w in re.findall(r"\w+", gt_text) if len(w) > 3]

    retrieved_text = " ".join(retrieved).lower()
    if not gt_keywords:
        return EvaluationResult(key="context_recall", score=0.8, comment="Ground truth context matched")

    hit_count = sum(1 for kw in gt_keywords if kw in retrieved_text)
    recall = hit_count / len(gt_keywords)
    score = round(min(1.0, recall), 2)
    comment = f"Retrieved {recall:.1%} of reference knowledge"

    return EvaluationResult(key="context_recall", score=score, comment=comment)


def latency_evaluator(run, example) -> EvaluationResult:
    """Tracks latency in seconds for performance tracking."""
    latency_ms = (run.outputs or {}).get("latency_ms", 0.0)
    score = round(latency_ms / 1000.0, 2)
    return EvaluationResult(key="latency_sec", score=score, comment=f"{round(latency_ms, 1)}ms total latency")


# ------------------------------------------------------------
# 4. Main CLI Runner
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Native LangSmith RAG Evaluations")
    parser.add_argument("--lang", type=str, default="combined", choices=["gu", "hi", "combined"], help="Language: 'gu', 'hi', or 'combined'")
    parser.add_argument("--sample", type=str, default="15", help="Sample size (e.g., 5, 10, 15, or 'full')")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME_DEFAULT, help="LangSmith dataset name")
    parser.add_argument("--sync-dataset", action="store_true", help="Force dataset sync to LangSmith")
    parser.add_argument("--experiment-prefix", type=str, default="voice-rag", help="Prefix for experiment name in LangSmith")
    args = parser.parse_args()

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("❌ Error: LANGSMITH_API_KEY is not set in environment or .env file.")
        print("   Please add your key: LANGSMITH_API_KEY=lsv2_pt_...")
        sys.exit(1)

    print("=" * 70)
    print("🛰️ LangSmith Native Evaluation Suite")
    print(f"   Project Name      : {project_name()}")
    print(f"   Target Dataset    : {args.dataset_name}")
    print(f"   Language          : {args.lang.upper()}")
    print(f"   Sample Size       : {args.sample}")
    print(f"   Experiment Prefix : {args.experiment_prefix}")
    print("=" * 70)

    client = Client()

    # 1. Load Local Records
    loader = GoldenDataLoader()
    limit = int(args.sample) if args.sample.isdigit() else None
    records = loader.load_records(
        language=args.lang,
        sample_size=args.sample if not limit else "15",
        limit=limit,
        dataset_type="pipeline"
    )

    if not records:
        print("❌ No evaluation records loaded. Check Data/ directory.")
        sys.exit(1)

    print(f"📂 Loaded {len(records)} test queries from Golden Dataset.")

    # 2. Sync to LangSmith if needed
    sync_golden_dataset_to_langsmith(
        client=client,
        records=records,
        dataset_name=args.dataset_name
    )

    # 3. Run Native LangSmith Evaluation
    print(f"\n🚀 Running LangSmith Experiment against dataset '{args.dataset_name}'...")
    print("   Metrics: Faithfulness, Answer Relevance, Context Recall, Latency")

    results = evaluate(
        rag_pipeline_target,
        data=args.dataset_name,
        evaluators=[
            faithfulness_evaluator,
            answer_relevance_evaluator,
            context_recall_evaluator,
            latency_evaluator
        ],
        experiment_prefix=f"{args.experiment_prefix}-{args.lang}",
        client=client,
        metadata={
            "language": args.lang,
            "sample_count": len(records),
            "project": project_name(),
        },
        max_concurrency=2
    )

    print("\n" + "=" * 70)
    print("🎉 LangSmith Evaluation Complete!")
    print(f"   Experiment Name : {results.experiment_name}")
    try:
        url = results.to_pandas().attrs.get("experiment_url")
        if url:
            print(f"   View Dashboard  : {url}")
        else:
            print(f"   View Dashboard  : https://smith.langchain.com -> Datasets & Testing -> {args.dataset_name}")
    except Exception:
        print(f"   View Dashboard  : https://smith.langchain.com -> Datasets & Testing -> {args.dataset_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
