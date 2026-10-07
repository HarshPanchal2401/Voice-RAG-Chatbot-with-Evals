"""
Golden Evaluation Data Loader
=============================
``--sample`` always decides which questions are evaluated (no metric-specific file can
override it):

    300 (default) / any integer N
        The pre-built stratified eval set ``eval_set_<N>`` when it exists
        (Data/gujarati|hindi/eval_set_<N>.jsonl, Data/combined/eval_set_<N>_paired.jsonl),
        otherwise N distinct questions are sampled on the fly from
        ``golden_dataset_full.jsonl`` with the same stratified, language-paired sampler
        (seeded). For ``combined`` N is the number of distinct questions, each evaluated
        in Gujarati AND Hindi (2N records).
    full
        Every record of ``golden_dataset_full.jsonl``.
    legacy100 / legacy500
        The old per-language ``golden_dataset_sample_<n>.jsonl`` files (gu / hi only).

``--dataset PATH`` loads an explicit JSONL file (e.g. a snapshot for ``--use-cached``).
Unknown languages and missing files raise errors - nothing silently falls back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation.sampling import language_of, sample_eval_records, select_records

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
LANG_FOLDERS = {"gu": "gujarati", "hi": "hindi"}
LANG_ALIASES = {"gu": "gu", "gujarati": "gu", "hi": "hi", "hindi": "hi", "combined": "combined", "all": "combined"}
DEFAULT_SAMPLE = "300"


def normalize_language(language: str) -> str:
    key = (language or "").strip().lower()
    if key not in LANG_ALIASES:
        raise ValueError(f"Unknown language {language!r}. Use one of: gu, hi, combined.")
    return LANG_ALIASES[key]


def eval_set_path(data_dir: Path, language: str, n: int) -> Path:
    lang = normalize_language(language)
    if lang == "combined":
        return data_dir / "combined" / f"eval_set_{n}_paired.jsonl"
    return data_dir / LANG_FOLDERS[lang] / f"eval_set_{n}.jsonl"


def read_jsonl(path: Path, default_language: Optional[str] = None) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lang = language_of(item, default_language)
            if lang not in ("gu", "hi"):
                raise ValueError(f"{path}:{i}: record has no usable language tag (got {lang!r})")
            item["language"] = lang
            records.append(item)
    return records


def _folder_language(path: Path) -> Optional[str]:
    parts = {p.lower() for p in path.parts}
    if "gujarati" in parts:
        return "gu"
    if "hindi" in parts:
        return "hi"
    return None


class GoldenDataLoader:
    """Loads and slices evaluation datasets. ``self.sources`` lists the files actually read."""

    def __init__(self, data_dir: Optional[Path] = None, seed: int = 42):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.seed = seed
        self.sources: List[Path] = []
        self.description: str = ""

    def full_path(self, lang: str) -> Path:
        return self.data_dir / LANG_FOLDERS[lang] / "golden_dataset_full.jsonl"

    def _require(self, path: Path, hint: str = "") -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}. {hint}".strip())
        self.sources.append(path)
        return path

    def load_file(self, path: Path | str) -> List[Dict[str, Any]]:
        p = self._require(Path(path))
        self.description = f"file {p.name}"
        return read_jsonl(p, _folder_language(p))

    def load_records(
        self,
        language: str = "combined",
        sample_size: str | int = DEFAULT_SAMPLE,
        limit: Optional[int] = None,
        query_type: Optional[str] = None,
        dataset_type: Optional[str] = None,  # accepted for backward compatibility; ignored
        drop_context_dependent: bool = True,
        dataset_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.sources = []
        if dataset_path:
            records = self.load_file(dataset_path)
        else:
            lang = normalize_language(language)
            langs: Sequence[str] = ("gu", "hi") if lang == "combined" else (lang,)
            sz = str(sample_size).strip().lower()
            if sz in ("full", "all"):
                records = []
                for lg in langs:
                    records.extend(read_jsonl(self._require(self.full_path(lg)), lg))
                self.description = "full golden dataset"
            elif sz.startswith("legacy"):
                n = sz.replace("legacy", "").lstrip("_")
                if lang == "combined":
                    raise ValueError("legacy samples exist only per language (--lang gu or --lang hi).")
                p = self._require(self.data_dir / LANG_FOLDERS[lang] / f"golden_dataset_sample_{n}.jsonl")
                records = read_jsonl(p, lang)
                self.description = f"legacy sample {n}"
            elif sz.isdigit() and int(sz) > 0:
                n = int(sz)
                prebuilt = eval_set_path(self.data_dir, lang, n)
                if prebuilt.exists():
                    records = read_jsonl(self._require(prebuilt), None if lang == "combined" else lang)
                    self.description = f"pre-built stratified eval set ({n} questions)"
                else:
                    by_lang = {lg: read_jsonl(self._require(self.full_path(lg)), lg) for lg in langs}
                    res = sample_eval_records(by_lang, n, seed=self.seed, drop_context_dependent=drop_context_dependent)
                    records = []
                    by_qid = {lg: {r["query_id"]: r for r in res["records"][lg]} for lg in langs}
                    for qid in res["query_ids"]:
                        for lg in langs:
                            records.append(by_qid[lg][qid])
                    self.description = f"on-the-fly stratified sample of {n} questions (seed={self.seed})"
            else:
                raise ValueError(f"Invalid --sample {sample_size!r}: use an integer, 'full', 'legacy100' or 'legacy500'.")
        if query_type:
            records = [r for r in records if str(r.get("query_type", "")).upper() == query_type.upper()]
        if limit:
            # Limit by distinct questions so Gujarati/Hindi pairs stay together.
            keep, seen = [], []
            for r in records:
                qid = r.get("query_id")
                if qid not in seen:
                    if len(seen) >= limit:
                        continue
                    seen.append(qid)
                keep.append(r)
            records = keep
        return records


def paired_order(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ids = []
    for r in records:
        if r.get("query_id") not in ids:
            ids.append(r.get("query_id"))
    return select_records(records, ids)
