"""
Evaluation Report Generator
===========================
Formats evaluation metrics into:
1. Rich terminal tables
2. Structured Markdown summary reports
3. Machine-readable JSON scorecards
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ReportGenerator:
    """Generates visual and persistent evaluation scorecards."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).resolve().parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_terminal_summary(self, eval_data: Dict[str, Any]):
        """Prints a clean ASCII table summary in the terminal."""
        component = eval_data.get("component", "unknown").upper()
        queries_count = eval_data.get("evaluated_queries", 0)
        aggs = eval_data.get("aggregates", {})

        print("\n" + "=" * 70)
        print(f"📊 VOICE RAG COMPONENT EVALUATION REPORT: [{component}]")
        print(f"   Evaluated Queries: {queries_count} | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if component in ("RETRIEVER", "RETRIEVER_RERANKED"):
            ov = aggs.get("overall_averages", aggs)
            print(f"  🎯 Contextual Recall    : {ov.get('contextual_recall', 0.0):.4f} / 1.0000")
            print(f"  🔍 Contextual Precision : {ov.get('contextual_precision', 0.0):.4f} / 1.0000")
            print(f"  ⏱️ Total Retrieval Lat  : {ov.get('retrieval_total_ms', 0.0):.2f} ms")

            # Breakdown by Query Type
            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                print("\n  📌 Breakdown by Query Type:")
                print("  " + "-" * 56)
                print(f"  {'Query Type':<15} | {'Count':<6} | {'Ctx Recall':<12} | {'Ctx Precision':<14}")
                print("  " + "-" * 56)
                for qt, d in by_type.items():
                    print(f"  {qt:<15} | {d['count']:<6} | {d['contextual_recall']:<12.4f} | {d['contextual_precision']:<14.4f}")

            # Breakdown by Language
            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                print("\n  🌐 Breakdown by Language:")
                print("  " + "-" * 56)
                print(f"  {'Language':<15} | {'Count':<6} | {'Ctx Recall':<12} | {'Ctx Precision':<14}")
                print("  " + "-" * 56)
                for lg, d in by_lang.items():
                    lang_name = "Gujarati" if lg == "gu" else ("Hindi" if lg == "hi" else lg)
                    print(f"  {lang_name:<15} | {d['count']:<6} | {d['contextual_recall']:<12.4f} | {d['contextual_precision']:<14.4f}")

        elif component in ("GENERATOR", "GENERATOR_RERANKED"):
            ov = aggs.get("overall_averages", aggs)
            mode = eval_data.get("mode", "oracle").upper()
            print(f"  Mode                    : {mode} Context")
            print(f"  🛡️ Faithfulness         : {ov.get('faithfulness', 0.0):.4f} / 1.0000")
            print(f"  🎯 Answer Relevancy     : {ov.get('answer_relevance', 0.0):.4f} / 1.0000")
            print(f"  ⏱️ LLM Generation Time  : {ov.get('llm_ms', 0.0):.2f} ms")
            print(f"  🚀 Time To First Token  : {ov.get('ttft_ms', 0.0):.2f} ms")

            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                print("\n  📌 Breakdown by Query Type:")
                print("  " + "-" * 56)
                print(f"  {'Query Type':<15} | {'Count':<6} | {'Faithfulness':<14} | {'Relevancy':<12}")
                print("  " + "-" * 56)
                for qt, d in by_type.items():
                    print(f"  {qt:<15} | {d['count']:<6} | {d['faithfulness']:<14.4f} | {d['answer_relevance']:<12.4f}")

            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                print("\n  🌐 Breakdown by Language:")
                print("  " + "-" * 56)
                print(f"  {'Language':<15} | {'Count':<6} | {'Faithfulness':<14} | {'Relevancy':<12}")
                print("  " + "-" * 56)
                for lg, d in by_lang.items():
                    lang_name = "Gujarati" if lg == "gu" else ("Hindi" if lg == "hi" else lg)
                    print(f"  {lang_name:<15} | {d['count']:<6} | {d['faithfulness']:<14.4f} | {d['answer_relevance']:<12.4f}")

        elif component in ("ALL", "PIPELINE"):
            ov = aggs.get("overall_averages", aggs)
            print(f"  🏆 Composite Pipeline Score : {ov.get('composite_score', 0.0):.4f} / 1.0000")
            print(f"  🛡️ Faithfulness             : {ov.get('faithfulness', 0.0):.4f} / 1.0000")
            print(f"  🎯 Answer Relevancy         : {ov.get('answer_relevance', 0.0):.4f} / 1.0000")
            print(f"  🔍 Context Relevancy        : {ov.get('context_relevance', 0.0):.4f} / 1.0000")
            print(f"  ⏱️ Total Pipeline Latency   : {ov.get('total_pipeline_ms', ov.get('average_latency_ms', 0.0)):.2f} ms")
            if "retrieval_total_ms" in ov:
                print(f"     ├── Retrieval Latency    : {ov.get('retrieval_total_ms', 0.0):.2f} ms")
                print(f"     └── LLM Generation Time  : {ov.get('llm_generation_ms', 0.0):.2f} ms")

            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                print("\n  📌 Breakdown by Query Type:")
                print("  " + "-" * 72)
                print(f"  {'Query Type':<14} | {'Count':<5} | {'Composite':<10} | {'Faithful':<10} | {'Relevancy':<10} | {'Ctx Relev':<10}")
                print("  " + "-" * 72)
                for qt, d in by_type.items():
                    print(f"  {qt:<14} | {d['count']:<5} | {d.get('composite_score', 0.0):<10.4f} | {d.get('faithfulness', 0.0):<10.4f} | {d.get('answer_relevance', 0.0):<10.4f} | {d.get('context_relevance', 0.0):<10.4f}")

            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                print("\n  🌐 Breakdown by Language:")
                print("  " + "-" * 72)
                print(f"  {'Language':<14} | {'Count':<5} | {'Composite':<10} | {'Faithful':<10} | {'Relevancy':<10} | {'Ctx Relev':<10}")
                print("  " + "-" * 72)
                for lg, d in by_lang.items():
                    lang_name = "Gujarati (gu)" if lg == "gu" else ("Hindi (hi)" if lg == "hi" else lg)
                    print(f"  {lang_name:<14} | {d['count']:<5} | {d.get('composite_score', 0.0):<10.4f} | {d.get('faithfulness', 0.0):<10.4f} | {d.get('answer_relevance', 0.0):<10.4f} | {d.get('context_relevance', 0.0):<10.4f}")

        elif component in ("APPLICATION", "APP"):
            ov = aggs.get("overall_averages", aggs)
            print(f"  🏆 Composite App Score      : {ov.get('composite_score', 0.0):.4f} / 1.0000")
            print(f"  🎯 Answer Correctness       : {ov.get('correctness', 0.0):.4f} / 1.0000")
            print(f"  📚 Answer Completeness      : {ov.get('completeness', 0.0):.4f} / 1.0000")
            print(f"  🛡️ Toxicity (Civility)      : {ov.get('toxicity', 0.0):.4f} / 1.0000")

            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                print("\n  📌 Breakdown by Query Type:")
                print("  " + "-" * 72)
                print(f"  {'Query Type':<14} | {'Count':<5} | {'Composite':<10} | {'Correctness':<12} | {'Complete':<10} | {'Toxicity':<10}")
                print("  " + "-" * 72)
                for qt, d in by_type.items():
                    print(f"  {qt:<14} | {d['count']:<5} | {d.get('composite_score', 0.0):<10.4f} | {d.get('correctness', 0.0):<12.4f} | {d.get('completeness', 0.0):<10.4f} | {d.get('toxicity', 0.0):<10.4f}")

            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                print("\n  🌐 Breakdown by Language:")
                print("  " + "-" * 72)
                print(f"  {'Language':<14} | {'Count':<5} | {'Composite':<10} | {'Correctness':<12} | {'Complete':<10} | {'Toxicity':<10}")
                print("  " + "-" * 72)
                for lg, d in by_lang.items():
                    lang_name = "Gujarati (gu)" if lg == "gu" else ("Hindi (hi)" if lg == "hi" else lg)
                    print(f"  {lang_name:<14} | {d['count']:<5} | {d.get('composite_score', 0.0):<10.4f} | {d.get('correctness', 0.0):<12.4f} | {d.get('completeness', 0.0):<10.4f} | {d.get('toxicity', 0.0):<10.4f}")

        print("=" * 70 + "\n")

    def save_reports(self, eval_data: Dict[str, Any], tag: str = "eval") -> Dict[str, Path]:
        """Saves evaluation results as both JSON scorecard and Markdown report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comp = eval_data.get("component", "report").lower()

        # 1. Save JSON
        json_path = self.output_dir / f"{comp}_{tag}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

        # 2. Save Markdown
        md_path = self.output_dir / f"{comp}_{tag}_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._format_markdown(eval_data))

        print(f"💾 Report saved to: {md_path}")
        print(f"💾 Raw JSON saved to: {json_path}")
        return {"json": json_path, "markdown": md_path}

    def _format_markdown(self, eval_data: Dict[str, Any]) -> str:
        comp = eval_data.get("component", "Report").capitalize()
        n = eval_data.get("evaluated_queries", 0)
        aggs = eval_data.get("aggregates", {})
        ov = aggs.get("overall_averages", aggs)

        lines = [
            f"# Voice RAG Evaluation Report: {comp}",
            f"**Evaluated Queries:** {n} | **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Summary Metrics\n",
            "| Metric | Score | Target Threshold | Status |",
            "| :--- | :--- | :--- | :--- |"
        ]

        if comp.upper() in ("RETRIEVER", "RETRIEVER_RERANKED"):
            lines.extend([
                f"| **Contextual Recall** | {ov.get('contextual_recall', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('contextual_recall', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Contextual Precision** | {ov.get('contextual_precision', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('contextual_precision', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Total Retrieval Latency** | {ov.get('retrieval_total_ms', 0.0):.2f} ms | $< 100$ ms | {'⚡ FAST' if ov.get('retrieval_total_ms', 0) < 100 else '🐢 MODERATE'} |"
            ])

            # Query type breakdown
            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                lines.extend([
                    "\n## Breakdown by Query Type\n",
                    "| Query Type | Count | Contextual Recall | Contextual Precision |",
                    "| :--- | :---: | :---: | :---: |"
                ])
                for qt, d in by_type.items():
                    lines.append(f"| **{qt}** | {d['count']} | {d['contextual_recall']:.4f} | {d['contextual_precision']:.4f} |")

            # Language breakdown
            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                lines.extend([
                    "\n## Breakdown by Language\n",
                    "| Language | Count | Contextual Recall | Contextual Precision |",
                    "| :--- | :---: | :---: | :---: |"
                ])
                for lg, d in by_lang.items():
                    lang_name = "Gujarati (gu)" if lg == "gu" else ("Hindi (hi)" if lg == "hi" else lg)
                    lines.append(f"| **{lang_name}** | {d['count']} | {d['contextual_recall']:.4f} | {d['contextual_precision']:.4f} |")

        elif comp.upper() in ("GENERATOR", "GENERATOR_RERANKED"):
            lines.extend([
                f"| **Faithfulness** | {ov.get('faithfulness', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('faithfulness', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Answer Relevancy** | {ov.get('answer_relevance', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('answer_relevance', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Generation Latency** | {ov.get('llm_ms', 0.0):.2f} ms | $< 2500$ ms | ⚡ |",
                f"| **Time To First Token** | {ov.get('ttft_ms', 0.0):.2f} ms | $< 500$ ms | ⚡ |"
            ])

            # Query type breakdown
            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                lines.extend([
                    "\n## Breakdown by Query Type\n",
                    "| Query Type | Count | Faithfulness | Answer Relevancy |",
                    "| :--- | :---: | :---: | :---: |"
                ])
                for qt, d in by_type.items():
                    lines.append(f"| **{qt}** | {d['count']} | {d['faithfulness']:.4f} | {d['answer_relevance']:.4f} |")

            # Language breakdown
            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                lines.extend([
                    "\n## Breakdown by Language\n",
                    "| Language | Count | Faithfulness | Answer Relevancy |",
                    "| :--- | :---: | :---: | :---: |"
                ])
                for lg, d in by_lang.items():
                    lang_name = "Gujarati (gu)" if lg == "gu" else ("Hindi (hi)" if lg == "hi" else lg)
                    lines.append(f"| **{lang_name}** | {d['count']} | {d['faithfulness']:.4f} | {d['answer_relevance']:.4f} |")

        elif comp.upper() in ("APPLICATION", "APP"):
            lines.extend([
                f"| **Composite Score** | {ov.get('composite_score', 0.0):.4f} | $\\ge 0.75$ | {'✅ PASS' if ov.get('composite_score', 0) >= 0.75 else '⚠️ REVIEW'} |",
                f"| **Correctness** | {ov.get('correctness', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('correctness', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Completeness** | {ov.get('completeness', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('completeness', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Toxicity (Safety)** | {ov.get('toxicity', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('toxicity', 0) >= 0.7 else '⚠️ REVIEW'} |"
            ])

            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                lines.extend([
                    "\n## Breakdown by Query Type\n",
                    "| Query Type | Count | Composite | Correctness | Completeness | Toxicity |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |"
                ])
                for qt, d in by_type.items():
                    lines.append(f"| **{qt}** | {d['count']} | {d.get('composite_score', 0.0):.4f} | {d.get('correctness', 0.0):.4f} | {d.get('completeness', 0.0):.4f} | {d.get('toxicity', 0.0):.4f} |")

            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                lines.extend([
                    "\n## Breakdown by Language\n",
                    "| Language | Count | Composite | Correctness | Completeness | Toxicity |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |"
                ])
                for lg, d in by_lang.items():
                    lang_name = "Gujarati (gu)" if lg == "gu" else ("Hindi (hi)" if lg == "hi" else lg)
                    lines.append(f"| **{lang_name}** | {d['count']} | {d.get('composite_score', 0.0):.4f} | {d.get('correctness', 0.0):.4f} | {d.get('completeness', 0.0):.4f} | {d.get('toxicity', 0.0):.4f} |")

            details = eval_data.get("details", [])
            if details:
                lines.extend([
                    "\n## Sample Q&A Overview Table\n",
                    "| # | Question | Language | Correctness | Completeness | Toxicity | Generated Answer Preview |",
                    "| :-: | :--- | :---: | :---: | :---: | :---: | :--- |"
                ])
                for idx, item in enumerate(details, start=1):
                    q = item.get("query", "").replace("\n", " ")[:35]
                    lang = item.get("language", "")
                    m = item.get("metrics", {})
                    ans = item.get("actual_output", "").replace("\n", " ")[:60]
                    lines.append(f"| {idx} | {q}... | {lang} | {m.get('correctness', 0.0):.2f} | {m.get('completeness', 0.0):.2f} | {m.get('toxicity', 0.0):.2f} | {ans}... |")

                lines.extend([
                    "\n## Detailed Sample Q&A Answers & Metric Evaluations\n"
                ])
                for idx, item in enumerate(details, start=1):
                    q_id = item.get("query_id", "N/A")
                    q_type = item.get("query_type", "N/A")
                    lang = item.get("language", "")
                    lang_name = "Gujarati (gu)" if lang == "gu" else ("Hindi (hi)" if lang == "hi" else lang)
                    m = item.get("metrics", {})
                    corr = m.get("correctness", 0.0)
                    corr_r = m.get("correctness_reason", "N/A")
                    comp_val = m.get("completeness", 0.0)
                    comp_r = m.get("completeness_reason", "N/A")
                    toxi = m.get("toxicity", 0.0)
                    toxi_r = m.get("toxicity_reason", "N/A")

                    lines.extend([
                        f"### Query {idx} (ID: `{q_id}`) — {lang_name} [{q_type}]",
                        f"- **Question:** {item.get('query', '')}",
                        f"- **Generated Answer:**",
                        f"  > {item.get('actual_output', '').strip()}",
                        f"- **Expected (Ground Truth) Answer:**",
                        f"  > {item.get('expected_output', '').strip()}",
                        f"- **Metrics & Reasoning:**",
                        f"  - **Correctness:** `{corr:.4f}` — {corr_r}",
                        f"  - **Completeness:** `{comp_val:.4f}` — {comp_r}",
                        f"  - **Toxicity (Safety):** `{toxi:.4f}` — {toxi_r}",
                        ""
                    ])

        else:
            # PIPELINE / ALL
            lines.extend([
                f"| **Composite Score** | {ov.get('composite_score', 0.0):.4f} | $\\ge 0.75$ | {'✅ PASS' if ov.get('composite_score', 0) >= 0.75 else '⚠️ REVIEW'} |",
                f"| **Faithfulness** | {ov.get('faithfulness', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('faithfulness', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Answer Relevancy** | {ov.get('answer_relevance', 0.0):.4f} | $\\ge 0.70$ | {'✅ PASS' if ov.get('answer_relevance', 0) >= 0.7 else '⚠️ REVIEW'} |",
                f"| **Context Relevancy** | {ov.get('context_relevance', 0.0):.4f} | $\\ge 0.50$ | {'✅ PASS' if ov.get('context_relevance', 0) >= 0.5 else '⚠️ REVIEW'} |",
                f"| **Total Pipeline Latency** | {ov.get('total_pipeline_ms', ov.get('average_latency_ms', 0.0)):.2f} ms | - | ⚡ |"
            ])

            by_type = aggs.get("breakdown_by_query_type", {})
            if by_type:
                lines.extend([
                    "\n## Breakdown by Query Type\n",
                    "| Query Type | Count | Composite | Faithfulness | Answer Relevancy | Context Relevancy |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |"
                ])
                for qt, d in by_type.items():
                    lines.append(f"| **{qt}** | {d['count']} | {d.get('composite_score', 0.0):.4f} | {d.get('faithfulness', 0.0):.4f} | {d.get('answer_relevance', 0.0):.4f} | {d.get('context_relevance', 0.0):.4f} |")

            by_lang = aggs.get("breakdown_by_language", {})
            if by_lang:
                lines.extend([
                    "\n## Breakdown by Language\n",
                    "| Language | Count | Composite | Faithfulness | Answer Relevancy | Context Relevancy |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |"
                ])
                for lg, d in by_lang.items():
                    lang_name = "Gujarati (gu)" if lg == "gu" else ("Hindi (hi)" if lg == "hi" else lg)
                    lines.append(f"| **{lang_name}** | {d['count']} | {d.get('composite_score', 0.0):.4f} | {d.get('faithfulness', 0.0):.4f} | {d.get('answer_relevance', 0.0):.4f} | {d.get('context_relevance', 0.0):.4f} |")

        return "\n".join(lines) + "\n"
