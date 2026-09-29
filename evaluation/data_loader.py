"""
Golden Evaluation Data Loader
==============================
Loads, validates, and filters Golden Dataset records for Gujarati, Hindi,
and Combined evaluation benchmarks. Supports metric-specific 15-sample datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from evaluation.config import DATA_DIR


class GoldenDataLoader:
    """Loads and slices evaluation datasets for component and end-to-end testing."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def resolve_dataset_path(
        self,
        language: str = "combined",
        sample_size: str = "15",
        dataset_type: str = "golden"
    ) -> Path:
        """
        Resolves path to target JSONL file.
        sample_size: '15', '100', '500', 'full'
        dataset_type: 'golden', 'retrieval', 'generator',
                      'contextual_recall', 'contextual_precision',
                      'faithfulness', 'answer_relevancy'
        """
        lang = language.lower().strip()
        folder = "combined" if lang == "combined" else ("gujarati" if lang in ("gu", "gujarati") else "hindi")
        base_folder = self.data_dir / folder

        # 1. Metric-specific datasets
        dt = dataset_type.lower().strip()
        metric_files = {
            "contextual_recall": "retrieval_golden_dataset.jsonl",
            "contextual_precision": "retrieval_golden_dataset.jsonl",
            "retrieval_eval": "retrieval_golden_dataset.jsonl",
            "retrieval": "retrieval_golden_dataset.jsonl",
            "faithfulness": "generator_golden_dataset.jsonl",
            "answer_relevance": "generator_golden_dataset.jsonl",
            "answer_relevancy": "generator_golden_dataset.jsonl",
            "generator": "generator_golden_dataset.jsonl",
            "generator_eval": "generator_golden_dataset.jsonl",
            "context_relevance": "pipeline_golden_dataset.jsonl",
            "context_relevancy": "pipeline_golden_dataset.jsonl",
            "contextual_relevance": "pipeline_golden_dataset.jsonl",
            "contextual_relevancy": "pipeline_golden_dataset.jsonl",
            "pipeline": "pipeline_golden_dataset.jsonl",
            "pipeline_eval": "pipeline_golden_dataset.jsonl",
        }
        if dt in metric_files:
            m_path = base_folder / metric_files[dt]
            if m_path.exists():
                return m_path
            if "pipe" in dt or "context_relevan" in dt or "contextual_relevan" in dt:
                alt_path = base_folder / "pipeline_golden_dataset.jsonl"
                if alt_path.exists():
                    return alt_path
            elif "retriev" in dt or "contextual" in dt:
                alt_path = base_folder / "retrieval_golden_dataset.jsonl"
                if alt_path.exists():
                    return alt_path
            elif "faith" in dt or "answer" in dt or "generator" in dt:
                alt_path = base_folder / "generator_golden_dataset.jsonl"
                if alt_path.exists():
                    return alt_path

        # 2. Sample size subsets
        sz = str(sample_size).lower().strip()
        if sz in ("15", "sample_15"):
            p15 = base_folder / "golden_dataset_sample_15.jsonl"
            if p15.exists():
                return p15
        elif sz in ("100", "sample_100"):
            p100 = base_folder / "golden_dataset_sample_100.jsonl"
            if p100.exists():
                return p100
        elif sz in ("500", "sample_500"):
            p500 = base_folder / "golden_dataset_sample_500.jsonl"
            if p500.exists():
                return p500
        elif sz in ("full", "all"):
            full_path = base_folder / "golden_dataset_full.jsonl"
            if full_path.exists():
                return full_path

        # Fallback hierarchy: 15 -> 100 -> 500 -> pipeline -> retrieval -> generator
        for cand in (
            "golden_dataset_sample_15.jsonl",
            "golden_dataset_sample_100.jsonl",
            "golden_dataset_sample_500.jsonl",
            "pipeline_golden_dataset.jsonl",
            "retrieval_golden_dataset.jsonl",
            "generator_golden_dataset.jsonl"
        ):
            cand_path = base_folder / cand
            if cand_path.exists():
                return cand_path

        return base_folder / "golden_dataset_sample_15.jsonl"

    def load_records(
        self,
        language: str = "combined",
        sample_size: str = "15",
        limit: Optional[int] = None,
        query_type: Optional[str] = None,
        dataset_type: str = "golden"
    ) -> List[Dict[str, Any]]:
        path = self.resolve_dataset_path(language=language, sample_size=sample_size, dataset_type=dataset_type)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset file not found at: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # Ensure language tag is present
                if "language" not in item:
                    item["language"] = "gu" if "gujarati" in str(path).lower() else "hi"

                # Filter by query_type if specified
                if query_type and item.get("query_type", "").upper() != query_type.upper():
                    continue

                records.append(item)
                if limit and len(records) >= limit:
                    break

        return records
