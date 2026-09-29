"""
Unified CLI Evaluation Runner
=============================
Run custom component-based evaluations on Voice RAG:
- Retriever Evaluation: Contextual Recall & Contextual Precision (+ Hit@K, MRR)
- Generator Evaluation: Faithfulness & Answer Relevancy (+ Semantic Sim, Token F1)
- Metric-specific Golden Datasets (15 samples total: 8 Gujarati + 7 Hindi)
- Pipeline Evaluation: End-to-end composite benchmark

Usage Examples:
  # 1. Run Contextual Recall evaluation on 15 total combined samples (8 GU + 7 HI)
  python -m evaluation.run_evaluation --metric contextual_recall

  # 2. Run Contextual Precision evaluation on 15 total combined samples (8 GU + 7 HI)
  python -m evaluation.run_evaluation --metric contextual_precision

  # 3. Run Faithfulness evaluation on 15 total combined samples (8 GU + 7 HI)
  python -m evaluation.run_evaluation --metric faithfulness

  # 4. Run Answer Relevancy evaluation on 15 total combined samples (8 GU + 7 HI)
  python -m evaluation.run_evaluation --metric answer_relevancy

  # 5. Fast Retriever evaluation on combined dataset (zero LLM token cost)
  python -m evaluation.run_evaluation --component retriever --sample 15 --no-llm-metrics

  # 6. Re-score the answers frozen inside the dataset files instead of running the live pipeline
  python -m evaluation.run_evaluation --component pipeline --use-cached

By default every record runs through the LIVE production pipeline (same code as the API).
With LANGSMITH_TRACING=true each record is traced to the "<LANGSMITH_PROJECT> - eval" project.
For LangSmith Datasets/Experiments use: python -m evaluation.langsmith_eval
"""

import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from evaluation.config import EvaluationConfig
from evaluation.client import RAGClient
from evaluation.data_loader import GoldenDataLoader
from evaluation.components.retriever_evaluator import RetrieverEvaluator
from evaluation.components.generator_evaluator import GeneratorEvaluator
from evaluation.components.application_evaluator import ApplicationEvaluator
from evaluation.pipeline_evaluator import PipelineEvaluator
from evaluation.report_generator import ReportGenerator
from core.tracing import tracing_enabled, project_name, flush as flush_traces


def parse_args():
    parser = argparse.ArgumentParser(description="Voice RAG Custom Component Evaluation Runner")
    parser.add_argument(
        "--component",
        type=str,
        choices=["retriever", "generator", "pipeline", "application", "all"],
        default="pipeline",
        help="Component to evaluate: 'retriever', 'generator', 'pipeline', 'application', or 'all' (default: pipeline)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["contextual_recall", "contextual_precision", "faithfulness", "answer_relevancy", "context_relevance", "correctness", "completeness", "toxicity"],
        default=None,
        help="Evaluate on dedicated 15-sample metric dataset ('contextual_recall', 'contextual_precision', 'faithfulness', 'answer_relevancy', 'context_relevance', 'correctness', 'completeness', 'toxicity')"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="combined",
        help="Language benchmark: 'gu' (Gujarati), 'hi' (Hindi), or 'combined' (default: combined)"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="15",
        help="Golden dataset sample: '15', '100', '500', or 'full' (default: 15)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (e.g. 5, 10, 15)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K context passages for retriever (default: 5)"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        default=True,
        help="Use ground-truth context for generator evaluation (Oracle Mode, default: True)"
    )
    parser.add_argument(
        "--no-oracle",
        dest="oracle",
        action="store_false",
        help="Use pipeline retrieved contexts for generator evaluation"
    )
    parser.add_argument(
        "--no-llm-metrics",
        action="store_true",
        help="Skip LLM judge calls for retriever (fast zero-token IR math metrics only)"
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default=None,
        help="Filter dataset by query_type ('DESCRIPTION', 'NUMERIC', 'ENTITY', 'LOCATION', 'PERSON')"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable multilingual re-ranker on retrieved candidates before evaluation"
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        default=False,
        help="Reuse retrieved contexts / answers / timings stored in the dataset instead of running the live pipeline"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save markdown and json reports to evaluation/reports/ (default: True)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Route metric shortcut to appropriate component
    if args.metric:
        if args.metric in ("contextual_recall", "contextual_precision"):
            args.component = "retriever"
            dataset_type = args.metric
        elif args.metric in ("faithfulness", "answer_relevancy"):
            args.component = "generator"
            dataset_type = args.metric
        elif args.metric in ("context_relevance", "context_relevancy"):
            args.component = "pipeline"
            dataset_type = "pipeline"
        elif args.metric in ("correctness", "completeness", "toxicity"):
            args.component = "application"
            dataset_type = "pipeline"
    else:
        dataset_type = "retrieval" if args.component == "retriever" else ("generator" if args.component == "generator" else "pipeline")

    print("=" * 70)
    print("🚀 Voice RAG Custom Evaluation Suite")
    comp_title = f"{args.component.upper()} (WITH RE-RANKER)" if getattr(args, "rerank", False) else args.component.upper()
    print(f"   Component   : {comp_title}")
    if args.metric:
        print(f"   Target Metric: {args.metric.upper()} (15 Total Samples: 8 GU + 7 HI)")
    print(f"   Language    : {args.lang.upper()}")
    print(f"   Sample Set  : {args.sample} (Limit: {args.limit or 'All in sample'})")
    if args.component in ("retriever", "all"):
        print(f"   Top-K       : {args.k}")
        print(f"   LLM Metrics : {'Disabled (IR Math Only)' if args.no_llm_metrics else 'Enabled (Contextual Recall/Precision)'}")
        if getattr(args, "rerank", False):
            print("   Re-Ranker   : Enabled (Groq Qwen 2.5 27B)")
    if args.component in ("generator", "all"):
        print(f"   Gen Mode    : {'Oracle (Gold Context)' if args.oracle else 'RAG (Retrieved Context)'}")
    print("=" * 70)

    # 1. Load Data
    loader = GoldenDataLoader()
    records = loader.load_records(
        language=args.lang,
        sample_size=args.sample,
        limit=args.limit,
        query_type=args.query_type,
        dataset_type=dataset_type
    )

    if not records:
        print("❌ No evaluation records found. Check dataset path or filters.")
        sys.exit(1)

    print(f"📦 Successfully loaded {len(records)} evaluation records!\n")

    # 2. Connect RAG Client (Live server or in-process fallback)
    client = RAGClient(base_url="http://localhost:8000")
    router = None
    if client.is_server_online():
        print("🔗 Connected to live RAG server at http://localhost:8000! (Zero duplicate RAM overhead)")
    else:
        print("ℹ️ Live RAG server not detected on http://localhost:8000. Initializing in-process router...")
        from pipeline.router import LanguageRouter
        router = LanguageRouter(verbose=False)
        client.fallback_router = router

    config = EvaluationConfig(language=args.lang, use_cached=args.use_cached)
    print(f"   Data Source : {'CACHED dataset snapshot' if args.use_cached else 'LIVE production pipeline'}")
    reporter = ReportGenerator()
    eval_result = None

    # 3. Execute Selected Component Evaluation
    if args.component == "retriever":
        evaluator = RetrieverEvaluator(
            router=router,
            client=client,
            config=config,
            run_llm_metrics=not args.no_llm_metrics,
            use_reranker=getattr(args, "rerank", False)
        )
        eval_result = evaluator.evaluate_dataset(records, k=args.k)
        if getattr(args, "rerank", False):
            eval_result["component"] = "retriever_reranked"

    elif args.component == "generator":
        evaluator = GeneratorEvaluator(
            router=router,
            client=client,
            config=config,
            use_reranker=getattr(args, "rerank", False)
        )
        eval_result = evaluator.evaluate_dataset(records, oracle_context=args.oracle)
        if getattr(args, "rerank", False):
            eval_result["component"] = "generator_reranked"

    elif args.component in ("all", "pipeline"):
        evaluator = PipelineEvaluator(
            router=router,
            client=client,
            config=config
        )
        eval_result = evaluator.evaluate_dataset(records, k=args.k)

    elif args.component == "application":
        evaluator = ApplicationEvaluator(
            router=router,
            client=client,
            config=config
        )
        eval_result = evaluator.evaluate_dataset(records)

    # 4. Generate Reports
    if eval_result:
        reporter.print_terminal_summary(eval_result)
        if args.save_report:
            metric_tag = f"_{args.metric}" if args.metric else ""
            tag = f"{args.lang}_{args.sample}{metric_tag}"
            reporter.save_reports(eval_result, tag=tag)


if __name__ == "__main__":
    if tracing_enabled():
        import langsmith
        eval_project = f"{project_name()} - eval"
        print(f"🛰️ LangSmith tracing ON -> project '{eval_project}'")
        with langsmith.tracing_context(project_name=eval_project, tags=["offline-eval"]):
            main()
        flush_traces()
    else:
        main()
