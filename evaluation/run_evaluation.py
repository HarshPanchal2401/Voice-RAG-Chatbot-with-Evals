"""
Unified CLI Evaluation Runner
=============================
Components:
  retriever    doc-key IR metrics (hit@k, recall@k, precision@k, MRR, nDCG) + contextual
               recall / precision (judge). ``--compare-rerank`` = paired rerank vs no-rerank.
  generator    faithfulness / answer relevance / token F1 / refusal metrics
               (``--oracle`` gold passages [default] or ``--no-oracle`` live pipeline)
  pipeline     end-to-end triad (faithfulness, answer relevance, context relevance) + IR + composite
  application  correctness / completeness (+ toxicity gate) + composite
  all          retriever, generator, pipeline and application (pipeline and application share one
               pipeline run)

Data: ``--sample 300`` (default) = the pre-built stratified eval set of 300 distinct questions
(per language; ``--lang combined`` evaluates each question in Gujarati AND Hindi). Build it with
``python -m evaluation.combine_datasets eval-sets --n 300``. ``--sample`` always wins.

Backend: ``--mode server`` (default; needs the API running, sends X-API-Key from RAG_API_KEY) or
``--mode inprocess`` (loads the full LanguageRouter: BGE-M3 + FAISS - needs several GB RAM).
There is no silent fallback between the two.

Examples:
  python -m evaluation.run_evaluation --component retriever --no-llm-metrics --lang combined
  python -m evaluation.run_evaluation --component retriever --compare-rerank --no-llm-metrics
  python -m evaluation.run_evaluation --metric faithfulness --lang gu --limit 50
  python -m evaluation.run_evaluation --component pipeline --use-cached --dataset Data/combined/snapshot_combined_300q_<ts>.jsonl

Exit codes: 0 ok, 2 configuration / data error or error rate above --max-error-rate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

METRIC_ROUTES = {
    "contextual_recall": ("retriever", ("contextual_recall",)),
    "contextual_precision": ("retriever", ("contextual_precision",)),
    "faithfulness": ("generator", ("faithfulness",)),
    "answer_relevancy": ("generator", ("answer_relevance",)),
    "answer_relevance": ("generator", ("answer_relevance",)),
    "context_relevance": ("pipeline", ("context_relevance",)),
    "correctness": ("application", ("answer_correctness",)),
    "completeness": ("application", ("completeness",)),
    "toxicity": ("application", ("toxicity_safety",)),
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--component", choices=["retriever", "generator", "pipeline", "application", "all"], default="pipeline")
    p.add_argument("--metric", choices=sorted(METRIC_ROUTES), default=None,
                   help="Evaluate a single judge metric (selects its component; uses the same --sample data)")
    p.add_argument("--lang", default="combined", help="gu | hi | combined (default: combined)")
    p.add_argument("--sample", default="300", help="integer N (stratified eval set of N questions; default 300), "
                                                     "'full', 'legacy100' or 'legacy500'")
    p.add_argument("--dataset", default=None, help="explicit JSONL file (e.g. a snapshot for --use-cached)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N distinct questions")
    p.add_argument("--query-type", default=None, help="filter by query_type (DESCRIPTION, NUMERIC, ENTITY, LOCATION, PERSON)")
    p.add_argument("--keep-context-dependent", action="store_true",
                   help="keep pronoun-only questions like 'Is she married?' when sampling on the fly")
    p.add_argument("--k", type=int, default=None, help="top-k passages (default: core.config FINAL_TOP_K or 5)")
    p.add_argument("--oracle", dest="oracle", action="store_true", default=True, help="generator: gold passages (default)")
    p.add_argument("--no-oracle", dest="oracle", action="store_false", help="generator: live pipeline retrieval")
    p.add_argument("--no-llm-metrics", action="store_true", help="skip all judge metrics (exact/lexical metrics only)")
    p.add_argument("--rerank", action="store_true", help="use the pipeline's reranker for retrieval")
    p.add_argument("--compare-rerank", action="store_true", help="retriever: paired rerank vs no-rerank comparison")
    p.add_argument("--use-cached", action="store_true", help="score a saved snapshot (needs --dataset)")
    p.add_argument("--allow-unknown-generator", action="store_true",
                   help="allow cached answers without a recorded generator model (warning in report)")
    p.add_argument("--mode", choices=["server", "inprocess"], default="server")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--judge-model", default=None, help="override RAG_JUDGE_MODEL (must differ from generator/reranker)")
    p.add_argument("--no-judge-cache", action="store_true", help="disable the on-disk judge cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=4, help="records scored concurrently by the judge")
    p.add_argument("--warmup", type=int, default=1, help="requests excluded from latency percentiles")
    p.add_argument("--max-error-rate", type=float, default=0.05, help="fail the run above this pipeline error rate")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for confidence intervals")
    p.add_argument("--save-report", action=argparse.BooleanOptionalAction, default=True,
                   help="write Markdown + JSON reports to evaluation/reports/ (default: on)")
    p.add_argument("--quiet", action="store_true", help="no progress bars")
    return p.parse_args(argv)


def _default_k() -> int:
    try:
        from core.config import FINAL_TOP_K

        return int(FINAL_TOP_K)
    except Exception:
        return 5


def _pipeline_settings() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import core.config as cc

        for name in ("FINAL_TOP_K", "RRF_TOP_N", "DENSE_TOP_N", "SPARSE_TOP_N", "RERANKER_BACKEND", "RERANK_FETCH_K",
                     "RERANK_MIN_SCORE", "CROSS_ENCODER_MODEL", "LLM_MAX_TOKENS", "BGE_QUERY_MAX_LENGTH"):
            if hasattr(cc, name):
                out[name] = getattr(cc, name)
    except Exception as e:  # pragma: no cover
        out["error"] = str(e)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    from evaluation.config import EvaluationConfig
    from evaluation.data_loader import GoldenDataLoader
    from evaluation.manifest import file_entry, run_manifest
    from evaluation.metrics.judge import ModelRoleError, check_model_roles
    from evaluation.metrics.retrieval_metrics import StaleSnapshotError
    from evaluation.report_generator import ReportGenerator
    from evaluation.components.common import CachedAnswerError

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    file_ts = time.strftime("%Y%m%d_%H%M%S")
    component = args.component
    only_metrics = None
    if args.metric:
        component, only_metrics = METRIC_ROUTES[args.metric]

    if args.use_cached and not args.dataset:
        print("ERROR: --use-cached needs --dataset <snapshot.jsonl> built with "
              "'python -m evaluation.combine_datasets snapshot ...' (live retrieval, doc keys, generator model).")
        return 2

    # 1. Data
    loader = GoldenDataLoader(seed=args.seed)
    try:
        records = loader.load_records(language=args.lang, sample_size=args.sample, limit=args.limit,
                                      query_type=args.query_type, drop_context_dependent=not args.keep_context_dependent,
                                      dataset_path=args.dataset)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading data: {e}")
        return 2
    if not records:
        print("ERROR: no evaluation records after filtering.")
        return 2

    # 2. Config + model roles
    try:
        config = EvaluationConfig(language=args.lang, mode=args.mode, base_url=args.base_url, use_cached=args.use_cached,
                                  allow_unknown_generator=args.allow_unknown_generator, default_k=args.k or _default_k(),
                                  seed=args.seed, concurrency=args.concurrency, warmup=args.warmup,
                                  max_error_rate=args.max_error_rate, judge_cache=not args.no_judge_cache,
                                  n_boot=args.n_boot, judge_model_override=args.judge_model)
        check_model_roles(config.roles)
    except ModelRoleError as e:
        print(f"ERROR (model roles): {e}")
        return 2

    roles = config.roles.as_dict()
    print("=" * 80)
    print("Voice RAG evaluation")
    print(f"  component : {component}" + (f" (metric: {args.metric})" if args.metric else ""))
    print(f"  data      : {loader.description} | {len(records)} records, "
          f"{len({r.get('query_id') for r in records})} distinct questions | lang={args.lang}")
    print(f"  models    : generator={roles['generator']} | judge={roles['judge']} | "
          f"reranker={roles['reranker_backend']}:{roles['reranker_model']}")
    print(f"  backend   : {'CACHED snapshot' if args.use_cached else args.mode} | k={config.default_k} | "
          f"rerank={args.rerank} | judge metrics={'off' if args.no_llm_metrics else 'on'}")
    print("=" * 80)

    # 3. Backend (explicit, no fallback)
    client = None
    needs_backend = not args.use_cached and not (component == "generator" and args.oracle)
    if needs_backend:
        from evaluation.client import RAGClient, RAGClientError, make_inprocess_router

        try:
            if args.mode == "inprocess":
                print("Loading in-process LanguageRouter (BGE-M3 + FAISS; several GB RAM)...")
                client = RAGClient(mode="inprocess", router=make_inprocess_router())
            else:
                client = RAGClient(mode="server", base_url=args.base_url)
            client.check()
        except RAGClientError as e:
            print(f"ERROR: {e}")
            return 2

    judge_off = args.no_llm_metrics
    show = not args.quiet
    results: List[Dict[str, Any]] = []
    shared_items = None
    judges = []

    def metrics_for(default):
        if judge_off:
            return ()
        return only_metrics if only_metrics else default

    try:
        if component in ("retriever", "all"):
            from evaluation.components.retriever_evaluator import RetrieverEvaluator
            from evaluation.metrics.retrieval_metrics import RETRIEVAL_LLM_METRICS

            ev = RetrieverEvaluator(client=client, config=config, llm_metrics=metrics_for(RETRIEVAL_LLM_METRICS),
                                    use_reranker=args.rerank, show_progress=show)
            judges.append(ev.judge)
            results.append(ev.evaluate_dataset(records, k=config.default_k, compare_rerank=args.compare_rerank))
        if component in ("generator", "all"):
            from evaluation.components.generator_evaluator import GeneratorEvaluator
            from evaluation.metrics.generation_metrics import GENERATION_LLM_METRICS

            ev = GeneratorEvaluator(client=client, config=config, use_reranker=args.rerank,
                                    llm_metrics=metrics_for(GENERATION_LLM_METRICS), show_progress=show)
            judges.append(ev.judge)
            results.append(ev.evaluate_dataset(records, oracle_context=args.oracle, k=config.default_k))
        if component in ("pipeline", "all"):
            from evaluation.pipeline_evaluator import PIPELINE_LLM_METRICS, PipelineEvaluator
            from evaluation.components.common import collect_generations

            ev = PipelineEvaluator(client=client, config=config, llm_metrics=metrics_for(PIPELINE_LLM_METRICS),
                                   use_reranker=args.rerank, show_progress=show)
            judges.append(ev.judge)
            shared_items = collect_generations(client, config, records, config.default_k, args.rerank, show)
            results.append(ev.evaluate_dataset(records, k=config.default_k, items=shared_items))
        if component in ("application", "all"):
            from evaluation.components.application_evaluator import ApplicationEvaluator
            from evaluation.metrics.application_metrics import APPLICATION_LLM_METRICS

            ev = ApplicationEvaluator(client=client, config=config, llm_metrics=metrics_for(APPLICATION_LLM_METRICS),
                                      use_reranker=args.rerank, show_progress=show)
            judges.append(ev.judge)
            results.append(ev.evaluate_dataset(records, k=config.default_k, items=shared_items))
    except (StaleSnapshotError, CachedAnswerError, ModelRoleError) as e:
        print(f"ERROR: {e}")
        return 2

    # 4. Manifest, status, reports
    exit_code = 0
    reporter = ReportGenerator()
    from evaluation.config import load_latency_budgets, load_lower_is_better, load_thresholds

    for res in results:
        judge = next((j for j in judges if j is not None), None)
        aggs = res.get("aggregates") or {}
        err_rate = aggs.get("error_rate") or 0.0
        res["run_status"] = "OK" if err_rate <= config.max_error_rate else (
            f"FAILED: pipeline error rate {err_rate:.1%} > max {config.max_error_rate:.1%}")
        if res["run_status"] != "OK":
            exit_code = 2
        res["thresholds"] = {"min": load_thresholds(), "max": load_lower_is_better(), "latency_p95_ms": load_latency_budgets()}
        res["manifest"] = run_manifest(
            timestamp=started,
            roles=roles,
            dataset_path=loader.sources[0] if loader.sources else None,
            mode="cached" if args.use_cached else args.mode,
            use_cached=args.use_cached,
            seed=args.seed,
            args=vars(args),
            judge=judge.summary() if judge is not None and not judge_off else None,
            extra={
                "dataset_sources": [file_entry(s) for s in loader.sources],
                "data_description": loader.description,
                "client": client.describe() if client is not None else None,
                "pipeline_settings": _pipeline_settings(),
                "allow_unknown_generator": args.allow_unknown_generator,
            },
        )
        reporter.print_terminal_summary(res)
        if args.save_report:
            sample_tag = "dataset" if args.dataset else str(args.sample)
            tag = f"{args.lang}_{sample_tag}" + (f"_{args.metric}" if args.metric else "") + ("_rerank" if args.rerank else "")
            reporter.save_reports(res, tag=tag, timestamp=file_ts)
    return exit_code


def _entry() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        return main()
    try:
        from core.tracing import flush as flush_traces, project_name, tracing_enabled
    except Exception:  # pragma: no cover
        return main()
    if tracing_enabled():
        import langsmith

        with langsmith.tracing_context(project_name=f"{project_name()} - eval", tags=["offline-eval"]):
            code = main()
        flush_traces()
        return code
    return main()


if __name__ == "__main__":
    sys.exit(_entry())
