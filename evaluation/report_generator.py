"""
Evaluation Report Generator
===========================
* Terminal summary, Markdown report and JSON scorecard for every evaluation run.
* Every metric is shown with its mean, bootstrap 95% CI, the n actually used, the
  judge success rate and a computed status (thresholds from ``evaluation.config`` /
  env, stored in the report):
      PASS          95% CI lower bound >= threshold
      INCONCLUSIVE  mean >= threshold but the CI reaches below it
      BELOW         mean < threshold
  (inverted for lower-is-better metrics such as refusal_rate).
* Latency: p50 / p95 excluding warm-up + cold start, status against p95 budgets.
* A manifest block (models, judge, dataset sha256, git commit, mode, versions, seed,
  timestamp) is embedded in every report.
* File names carry the real number of evaluated records: ``<component>_<tag>_n<N>_<ts>``.

Legacy reports (produced before the 2026-10-06 evaluation fixes) can be re-rendered
from their JSON with a prominent "not trustworthy" banner::

    python -m evaluation.report_generator --regenerate-legacy evaluation/reports/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

LEGACY_BANNER = (
    "> **⚠️ LEGACY REPORT - SCORES ARE NOT TRUSTWORTHY.** These numbers were produced before the "
    "evaluation fixes of 2026-10-06 and must not be used to judge the system: relevance was matched on "
    "non-unique passage ids (unrelated passages counted as hits), failed judge calls were silently replaced "
    "by hand-made heuristic scores, Gujarati/Hindi text was mis-tokenized, saved answers were judged by the "
    "same model that generated them, and the '15-sample' set contained only 8 distinct questions. "
    "This file was re-rendered from its JSON by `evaluation/report_generator.py`; the JSON is unchanged."
)

LANG_NAMES = {"gu": "Gujarati (gu)", "hi": "Hindi (hi)"}


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(s: Dict[str, Any]) -> str:
    if s.get("ci_low") is None or s.get("ci_high") is None:
        return "—"
    return f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def _thresholds(eval_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    th = eval_data.get("thresholds")
    if th:
        return th
    try:
        from evaluation.config import load_latency_budgets, load_lower_is_better, load_thresholds

        return {"min": load_thresholds(), "max": load_lower_is_better(), "latency_p95_ms": load_latency_budgets()}
    except Exception:
        return {"min": {}, "max": {}, "latency_p95_ms": {}}


def metric_status(name: str, s: Dict[str, Any], th: Dict[str, Dict[str, float]]) -> str:
    mean = s.get("mean")
    if mean is None:
        return "NO DATA"
    lo, hi = s.get("ci_low"), s.get("ci_high")
    status = "—"
    if name in (th.get("min") or {}):
        t = th["min"][name]
        if lo is not None and lo >= t:
            status = f"PASS (≥{t:g})"
        elif mean < t:
            status = f"BELOW (<{t:g})"
        else:
            status = f"INCONCLUSIVE (~{t:g})"
    elif name in (th.get("max") or {}):
        t = th["max"][name]
        if hi is not None and hi <= t:
            status = f"PASS (≤{t:g})"
        elif mean > t:
            status = f"ABOVE LIMIT (>{t:g})"
        else:
            status = f"INCONCLUSIVE (~{t:g})"
    jsr = s.get("judge_success_rate")
    if jsr is not None and jsr < 0.9:
        status += f" ⚠ judge ok {jsr:.0%}"
    return status


def latency_status(name: str, s: Dict[str, Any], th: Dict[str, Dict[str, float]]) -> str:
    budget = (th.get("latency_p95_ms") or {}).get(name)
    if budget is None or s.get("p95_ms") is None:
        return "—"
    return f"PASS (p95 ≤ {budget:g})" if s["p95_ms"] <= budget else f"SLOW (p95 > {budget:g})"


class ReportGenerator:
    """Generates terminal, Markdown and JSON scorecards."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ terminal
    def print_terminal_summary(self, eval_data: Dict[str, Any]) -> None:
        comp = str(eval_data.get("component", "unknown")).upper()
        aggs = eval_data.get("aggregates") or {}
        th = _thresholds(eval_data)
        man = eval_data.get("manifest") or {}
        print("\n" + "=" * 96)
        print(f"VOICE RAG EVALUATION: [{comp}]  records ok={aggs.get('n_ok')} errors={aggs.get('n_errors')} "
              f"distinct questions={aggs.get('distinct_query_ids')}")
        models = man.get("models") or {}
        if models:
            print(f"  generator={models.get('generator')} judge={models.get('judge')} "
                  f"reranker={models.get('reranker_backend')}:{models.get('reranker_model')} mode={man.get('mode')} "
                  f"cached={man.get('use_cached')}")
        print("=" * 96)
        print(f"  {'Metric':<30} {'Mean':>8} {'95% CI':>18} {'n used':>7} {'failed':>7} {'judge ok':>9}  Status")
        for name, s in (aggs.get("metrics") or {}).items():
            jsr = s.get("judge_success_rate")
            print(f"  {name:<30} {_fmt(s.get('mean')):>8} {_ci(s):>18} {s.get('n_used', 0):>7} {s.get('n_failed', 0):>7} "
                  f"{(f'{jsr:.0%}' if jsr is not None else '—'):>9}  {metric_status(name, s, th)}")
        lat = eval_data.get("latency") or {}
        if lat:
            print("\n  Latency (ms, warm-up excluded):")
            for name, s in lat.items():
                print(f"  {name:<22} p50={_fmt(s.get('p50_ms'), 1):>9} p95={_fmt(s.get('p95_ms'), 1):>9} "
                      f"cold={_fmt(s.get('cold_start_ms'), 1):>9} n={s.get('n')}  {latency_status(name, s, th)}")
        gate = eval_data.get("toxicity_gate")
        if gate:
            print(f"\n  Toxicity gate: {gate['status']} (flagged={gate['flagged_query_ids']}, unjudged={gate['n_unjudged']})")
        pc = (eval_data.get("paired_comparison") or {}).get("metrics") or {}
        if pc:
            print("\n  Paired comparison (rerank - no_rerank):")
            for m, s in pc.items():
                st = s.get("sign_test") or {}
                print(f"  {m:<26} diff={_fmt(s.get('mean_diff'))} CI=[{_fmt(s.get('ci_low'), 3)}, {_fmt(s.get('ci_high'), 3)}] "
                      f"p={_fmt(s.get('p_value'), 3)} wins rerank/base/ties={st.get('wins_b')}/{st.get('wins_a')}/{st.get('ties')} "
                      f"n={s.get('n_pairs')}")
        if eval_data.get("run_status"):
            print(f"\n  RUN STATUS: {eval_data['run_status']}")
        print("=" * 96 + "\n")

    # ------------------------------------------------------------ files
    def report_basename(self, eval_data: Dict[str, Any], tag: str, timestamp: Optional[str] = None) -> str:
        comp = str(eval_data.get("component", "report")).lower()
        n = (eval_data.get("aggregates") or {}).get("n_ok", eval_data.get("evaluated_queries", 0))
        ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
        return f"{comp}_{tag}_n{n}_{ts}"

    def save_reports(self, eval_data: Dict[str, Any], tag: str = "eval", timestamp: Optional[str] = None) -> Dict[str, Path]:
        base = self.report_basename(eval_data, tag, timestamp)
        json_path = self.output_dir / f"{base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False, default=str)
        md_path = self.output_dir / f"{base}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.format_markdown(eval_data))
        print(f"Report saved to: {md_path}")
        print(f"Raw JSON saved to: {json_path}")
        return {"json": json_path, "markdown": md_path}

    # ------------------------------------------------------------ markdown
    def format_markdown(self, eval_data: Dict[str, Any]) -> str:
        if "manifest" not in eval_data and "aggregates" in eval_data and "metrics" not in (eval_data.get("aggregates") or {}):
            return self.format_legacy_markdown(eval_data, eval_data.get("_source_name", ""))
        comp = str(eval_data.get("component", "report"))
        aggs = eval_data.get("aggregates") or {}
        th = _thresholds(eval_data)
        man = eval_data.get("manifest") or {}
        lines: List[str] = [f"# Voice RAG Evaluation Report: {comp}", ""]
        if eval_data.get("run_status") and eval_data["run_status"] != "OK":
            lines += [f"> **RUN STATUS: {eval_data['run_status']}**", ""]
        lines += [
            f"**Records evaluated:** {aggs.get('n_ok')} ok / {aggs.get('n_records')} total "
            f"(**errors:** {aggs.get('n_errors')}, error rate {_fmt(aggs.get('error_rate'))}) · "
            f"**distinct questions:** {aggs.get('distinct_query_ids')} · **k:** {eval_data.get('k', '—')} · "
            f"**mode:** {eval_data.get('mode', man.get('mode', '—'))}",
            "",
            "## Manifest",
            "",
            "```json",
            json.dumps(man, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## Summary metrics",
            "",
            "Mean over records where the metric was computed; failed judge calls are excluded (never replaced) "
            "and counted. 95% CI = percentile bootstrap clustered by query_id.",
            "",
            "| Metric | Mean | 95% CI | n used / records | failed | judge success | Status |",
            "| :--- | ---: | :---: | :---: | ---: | ---: | :--- |",
        ]
        for name, s in (aggs.get("metrics") or {}).items():
            jsr = s.get("judge_success_rate")
            lines.append(
                f"| **{name}** | {_fmt(s.get('mean'))} | {_ci(s)} | {s.get('n_used')} / {s.get('n_records')} | "
                f"{s.get('n_failed')} | {(f'{jsr:.0%}' if jsr is not None else '—')} | {metric_status(name, s, th)} |"
            )
        if eval_data.get("composite"):
            c = eval_data["composite"]
            lines += ["", f"Composite basis `{c.get('basis')}` weights: `{json.dumps(c.get('weights'))}` — {c.get('rule')}."]
        gate = eval_data.get("toxicity_gate")
        if gate:
            lines += ["", "## Toxicity gate", "",
                      f"**{gate['status']}** — {gate['rule']}. Judged: {gate['n_judged']}, unjudged: {gate['n_unjudged']}, "
                      f"flagged query_ids: {gate['flagged_query_ids'] or 'none'}. Direction: `{eval_data.get('toxicity_direction')}`."]
        lat = eval_data.get("latency") or {}
        if lat:
            lines += ["", "## Latency (ms)", "", "Warm-up request(s) excluded from percentiles; cold start shown separately.", "",
                      "| Stage | p50 | p95 | mean | cold start | n | Status |", "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |"]
            for name, s in lat.items():
                lines.append(f"| {name} | {_fmt(s.get('p50_ms'), 1)} | {_fmt(s.get('p95_ms'), 1)} | {_fmt(s.get('mean_ms'), 1)} | "
                             f"{_fmt(s.get('cold_start_ms'), 1)} | {s.get('n')} | {latency_status(name, s, th)} |")
        pc = eval_data.get("paired_comparison")
        if pc:
            lines += ["", "## Paired comparison: rerank vs no-rerank", "", pc.get("definition", ""), "",
                      "| Metric | no-rerank mean | rerank mean | diff (rerank − base) | 95% CI | bootstrap p | sign test (wins rerank/base/ties, p) | n pairs |",
                      "| :--- | ---: | ---: | ---: | :---: | ---: | :--- | ---: |"]
            arms = eval_data.get("arms") or {}
            for m, s in (pc.get("metrics") or {}).items():
                a = ((arms.get("no_rerank") or {}).get("aggregates") or {}).get("metrics", {}).get(m, {})
                b = ((arms.get("rerank") or {}).get("aggregates") or {}).get("metrics", {}).get(m, {})
                st = s.get("sign_test") or {}
                lines.append(f"| {m} | {_fmt(a.get('mean'))} | {_fmt(b.get('mean'))} | {_fmt(s.get('mean_diff'))} | "
                             f"[{_fmt(s.get('ci_low'), 3)}, {_fmt(s.get('ci_high'), 3)}] | {_fmt(s.get('p_value'), 3)} | "
                             f"{st.get('wins_b')}/{st.get('wins_a')}/{st.get('ties')}, p={_fmt(st.get('p_value'), 3)} | {s.get('n_pairs')} |")
        for key, title in (("breakdown_by_language", "Breakdown by language"), ("breakdown_by_query_type", "Breakdown by query type")):
            groups = aggs.get(key) or {}
            if not groups:
                continue
            names = list((aggs.get("metrics") or {}).keys())[:8]
            lines += ["", f"## {title}", "", "| Group | n | " + " | ".join(names) + " |",
                      "| :--- | ---: | " + " | ".join([":---:"] * len(names)) + " |"]
            for g, d in groups.items():
                cells = []
                for m in names:
                    s = (d.get("metrics") or {}).get(m) or {}
                    cells.append(f"{_fmt(s.get('mean'), 3)} {_ci(s)} (n={s.get('n_used', 0)})")
                lines.append(f"| {LANG_NAMES.get(g, g)} | {d.get('count')} | " + " | ".join(cells) + " |")
        errs = aggs.get("errors") or []
        if errs:
            lines += ["", "## Error records (excluded from all metrics)", "", "| query_id | lang | error |", "| :--- | :--- | :--- |"]
            for e in errs[:100]:
                lines.append(f"| {e.get('query_id')} | {e.get('language')} | {str(e.get('error')).replace('|', '/')[:200]} |")
        details = eval_data.get("details") or []
        if details and comp != "retriever_rerank_comparison":
            names = list((aggs.get("metrics") or {}).keys())[:6]
            lines += ["", "## Per-query results (first 60)", "",
                      "| # | query_id | lang | type | " + " | ".join(names) + " | failed metrics |",
                      "| ---: | :--- | :---: | :--- | " + " | ".join([":---:"] * len(names)) + " | :--- |"]
            for i, r in enumerate(details[:60], 1):
                sc = r.get("scores") or {}
                status = r.get("status")
                cells = [_fmt(sc.get(m), 3) for m in names] if status == "ok" else ["ERROR"] * len(names)
                lines.append(f"| {i} | {r.get('query_id')} | {r.get('language')} | {r.get('query_type')} | " + " | ".join(cells)
                             + f" | {', '.join(r.get('failed_metrics') or []) or '—'} |")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ legacy
    def format_legacy_markdown(self, eval_data: Dict[str, Any], source_name: str = "") -> str:
        comp = str(eval_data.get("component", "report"))
        n = eval_data.get("evaluated_queries", 0)
        details = eval_data.get("details") or []
        distinct = len({(d.get("query_id")) for d in details}) if details else None
        aggs = eval_data.get("aggregates") or {}
        ov = aggs.get("overall_averages") or {}
        lines = [f"# LEGACY Voice RAG Evaluation Report: {comp}", "", LEGACY_BANNER, "",
                 f"**Source JSON:** `{source_name}` · **records actually evaluated:** {n}"
                 + (f" · **distinct query_ids:** {distinct}" if distinct is not None else "")
                 + (" · **mode:** " + str(eval_data.get("mode")) if eval_data.get("mode") else "")
                 + (" · **k:** " + str(eval_data.get("k")) if eval_data.get("k") else ""),
                 "", "No manifest, no confidence intervals and no judge-failure accounting were recorded for this run.",
                 "", "## Recorded averages (untrustworthy)", "", "| Metric | Value |", "| :--- | ---: |"]
        for k, v in ov.items():
            lines.append(f"| {k} | {_fmt(v)} |")
        for key, title in (("breakdown_by_language", "By language"), ("breakdown_by_query_type", "By query type")):
            groups = aggs.get(key) or {}
            if not groups:
                continue
            cols = sorted({c for d in groups.values() for c in d.keys() if c != "count"})
            lines += ["", f"## {title} (untrustworthy)", "", "| Group | count | " + " | ".join(cols) + " |",
                      "| :--- | ---: | " + " | ".join(["---:"] * len(cols)) + " |"]
            for g, d in groups.items():
                lines.append(f"| {LANG_NAMES.get(g, g)} | {d.get('count')} | " + " | ".join(_fmt(d.get(c)) for c in cols) + " |")
        fb = 0
        for d in details:
            m = d.get("metrics") or {}
            if any("heuristic fallback" in str(v) for v in m.values()):
                fb += 1
        if details:
            lines += ["", f"Records whose stored reasons show a heuristic fallback (judge failure replaced by a made-up score): **{fb} / {len(details)}**."]
        return "\n".join(lines) + "\n"

    def regenerate_legacy(self, json_path: Path | str) -> Path:
        p = Path(json_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        md = p.with_suffix(".md")
        with open(md, "w", encoding="utf-8") as f:
            f.write(self.format_legacy_markdown(data, p.name))
        return md

    def render(self, json_path: Path | str) -> Path:
        p = Path(json_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source_name"] = p.name
        md = p.with_suffix(".md")
        with open(md, "w", encoding="utf-8") as f:
            f.write(self.format_markdown(data))
        return md

    # Backwards-compatible private name
    _format_markdown = format_markdown


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Re-render evaluation reports from their JSON (no API calls).")
    ap.add_argument("--regenerate-legacy", nargs="+", metavar="JSON", help="legacy report JSON files (globs ok)")
    ap.add_argument("--render", nargs="+", metavar="JSON", help="new-format report JSON files to re-render")
    args = ap.parse_args(argv)
    rg = ReportGenerator()
    for pattern in args.regenerate_legacy or []:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            print(f"legacy -> {rg.regenerate_legacy(path)}")
    for pattern in args.render or []:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            print(f"render -> {rg.render(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
