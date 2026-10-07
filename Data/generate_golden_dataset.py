"""
Unified Golden Evaluation Dataset Generator for Multi-Language Voice & Text RAG
================================================================================

A future-proof, unified generator for evaluating any language corpus (Gujarati, Hindi, 
Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, English, or any new language).

Key Features:
- 🌐 Universal Language Registry: Pre-configured for Indic & global languages with auto-fallback for any new language folder.
- 🔍 Dynamic Asset Discovery: Locates query files and SQLite databases without hardcoding filenames.
- 🛡️ Flexible Schema Adaptation: Handles varying column names (`query_id`, `chunk_id`, `text`, `answer`, `is_selected`).
- 🛑 Multilingual Unanswerable Filter: Detects negative/unanswerable patterns across languages.
- 📊 Stratified Subsampling & Diagnostics: Generates full + benchmark subsets (100, 500), stats, and README.

Usage:
  python generate_golden_dataset.py --lang hindi
  python generate_golden_dataset.py --lang gujarati
  python generate_golden_dataset.py --lang marathi
  python generate_golden_dataset.py --all
  python generate_golden_dataset.py --list
"""

import sys
import os
import json
import sqlite3
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# Universal Language Registry & Unanswerable Triggers
# ============================================================

LANGUAGE_REGISTRY: Dict[str, Dict[str, str]] = {
    # Indic Languages
    "gujarati": {"code": "gu", "name": "Gujarati", "locale": "gu-IN"},
    "gu": {"code": "gu", "name": "Gujarati", "locale": "gu-IN"},
    "hindi": {"code": "hi", "name": "Hindi", "locale": "hi-IN"},
    "hi": {"code": "hi", "name": "Hindi", "locale": "hi-IN"},
    "marathi": {"code": "mr", "name": "Marathi", "locale": "mr-IN"},
    "mr": {"code": "mr", "name": "Marathi", "locale": "mr-IN"},
    "bengali": {"code": "bn", "name": "Bengali", "locale": "bn-IN"},
    "bn": {"code": "bn", "name": "Bengali", "locale": "bn-IN"},
    "tamil": {"code": "ta", "name": "Tamil", "locale": "ta-IN"},
    "ta": {"code": "ta", "name": "Tamil", "locale": "ta-IN"},
    "telugu": {"code": "te", "name": "Telugu", "locale": "te-IN"},
    "te": {"code": "te", "name": "Telugu", "locale": "te-IN"},
    "kannada": {"code": "kn", "name": "Kannada", "locale": "kn-IN"},
    "kn": {"code": "kn", "name": "Kannada", "locale": "kn-IN"},
    "malayalam": {"code": "ml", "name": "Malayalam", "locale": "ml-IN"},
    "ml": {"code": "ml", "name": "Malayalam", "locale": "ml-IN"},
    "punjabi": {"code": "pa", "name": "Punjabi", "locale": "pa-IN"},
    "pa": {"code": "pa", "name": "Punjabi", "locale": "pa-IN"},
    "odia": {"code": "or", "name": "Odia", "locale": "or-IN"},
    "or": {"code": "or", "name": "Odia", "locale": "or-IN"},
    "assamese": {"code": "as", "name": "Assamese", "locale": "as-IN"},
    "as": {"code": "as", "name": "Assamese", "locale": "as-IN"},
    "urdu": {"code": "ur", "name": "Urdu", "locale": "ur-IN"},
    "ur": {"code": "ur", "name": "Urdu", "locale": "ur-IN"},
    "sanskrit": {"code": "sa", "name": "Sanskrit", "locale": "sa-IN"},
    "sa": {"code": "sa", "name": "Sanskrit", "locale": "sa-IN"},
    # International Languages
    "english": {"code": "en", "name": "English", "locale": "en-US"},
    "en": {"code": "en", "name": "English", "locale": "en-US"},
    "spanish": {"code": "es", "name": "Spanish", "locale": "es-ES"},
    "es": {"code": "es", "name": "Spanish", "locale": "es-ES"},
    "french": {"code": "fr", "name": "French", "locale": "fr-FR"},
    "fr": {"code": "fr", "name": "French", "locale": "fr-FR"},
    "german": {"code": "de", "name": "German", "locale": "de-DE"},
    "de": {"code": "de", "name": "German", "locale": "de-DE"},
}

MULTILINGUAL_UNANSWERABLE_TRIGGERS: Dict[str, List[str]] = {
    # Gujarati
    "gu": ["કોઈ જવાબ નથી", "કોઈ ઉત્તર નથી", "માહિતી ઉપલબ્ધ નથી", "નો જવાબ નથી"],
    "gujarati": ["કોઈ જવાબ નથી", "કોઈ ઉત્તર નથી", "માહિતી ઉપલબ્ધ નથી", "નો જવાબ નથી"],
    # Hindi
    "hi": ["कोई उत्तर नहीं", "कोई उत्तर नहीं मिला", "कोई जवाब नहीं", "उत्तर उपलब्ध नहीं", "जानकारी उपलब्ध नहीं"],
    "hindi": ["कोई उत्तर नहीं", "कोई उत्तर नहीं मिला", "कोई जवाब नहीं", "उत्तर उपलब्ध नहीं", "जानकारी उपलब्ध नहीं"],
    # Marathi
    "mr": ["काही उत्तर नाही", "उत्तर उपलब्ध नाही", "माहिती उपलब्ध नाही", "कोणतेही उत्तर नाही"],
    "marathi": ["काही उत्तर नाही", "उत्तर उपलब्ध नाही", "माहिती उपलब्ध नाही", "कोणतेही उत्तर नाही"],
    # Bengali
    "bn": ["কোন উত্তর নেই", "উত্তর পাওয়া যায়নি", "তথ্য পাওয়া যায়নি", "কোনো উত্তর নেই"],
    "bengali": ["কোন উত্তর নেই", "উত্তর পাওয়া যায়নি", "তথ্য পাওয়া যায়নি", "কোনো উত্তর নেই"],
    # Tamil
    "ta": ["பதில் இல்லை", "விடை கிடைக்கவில்லை", "தகவல் இல்லை"],
    "tamil": ["பதில் இல்லை", "விடை கிடைக்கவில்லை", "தகவல் இல்லை"],
    # Telugu
    "te": ["సమాధానం లేదు", "సమాచారం అందుబాటులో లేదు", "సమాధానం దొరకలేదు"],
    "telugu": ["సమాధానం లేదు", "సమాచారం అందుబాటులో లేదు", "సమాధానం దొరకలేదు"],
    # Kannada
    "kn": ["ಯಾವುದೇ ಉತ್ತರವಿಲ್ಲ", "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ", "ಉತ್ತರ ಸಿಗಲಿಲ್ಲ"],
    "kannada": ["ಯಾವುದೇ ಉತ್ತರವಿಲ್ಲ", "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ", "ಉತ್ತರ ಸಿಗಲಿಲ್ಲ"],
    # Malayalam
    "ml": ["ഉത്തരമില്ല", "വിവരം ലഭ്യമല്ല", "ഉത്തരം ലഭ്യമല്ല"],
    "malayalam": ["ഉത്തരമില്ല", "വിവരം ലഭ്യമല്ല", "ഉത്തരം ലഭ്യമല്ല"],
    # Punjabi
    "pa": ["ਕੋਈ ਜਵਾਬ ਨਹੀਂ", "ਕੋਈ ਉੱਤਰ ਨਹੀਂ", "ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ"],
    "punjabi": ["ਕੋਈ ਜਵਾਬ ਨਹੀਂ", "ਕੋਈ ਉੱਤਰ ਨਹੀਂ", "ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ"],
    # Odia
    "or": ["କୌଣସି ଉତ୍ତର ନାହିଁ", "ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ"],
    "odia": ["କୌଣସି ଉତ୍ତର ନାହିଁ", "ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ"],
    # Urdu
    "ur": ["کوئی جواب نہیں", "معلومات دستیاب نہیں"],
    "urdu": ["کوئی جواب نہیں", "معلومات دستیاب نہیں"],
    # English & Global
    "en": ["no answer", "no answer found", "not available", "unanswerable", "no information", "cannot be answered"],
    "english": ["no answer", "no answer found", "not available", "unanswerable", "no information", "cannot be answered"],
}

# Generic cross-lingual unanswerable tokens
GENERIC_UNANSWERABLE = ["no answer", "none", "n/a", "null", "no context", "not found", "unanswerable", "[]"]


def resolve_language_metadata(lang_dir_name: str, detected_lang_code: Optional[str] = None) -> Dict[str, str]:
    """
    Resolves standard language metadata (code, name, locale).
    If language is not in the registry, dynamically synthesizes metadata for any new language.
    """
    clean_name = lang_dir_name.strip().lower()
    
    # 1. Direct registry hit by name
    if clean_name in LANGUAGE_REGISTRY:
        return LANGUAGE_REGISTRY[clean_name]

    # 2. Registry hit by detected code
    if detected_lang_code and detected_lang_code.lower() in LANGUAGE_REGISTRY:
        return LANGUAGE_REGISTRY[detected_lang_code.lower()]

    # 3. Dynamic synthesis for arbitrary future language
    code = (detected_lang_code or clean_name[:2]).lower()
    formatted_name = lang_dir_name.replace("_", " ").replace("-", " ").title()
    locale = f"{code}-IN" if len(code) == 2 else f"{code}"

    return {
        "code": code,
        "name": formatted_name,
        "locale": locale
    }


def find_language_files(lang_dir: Path) -> Tuple[Path, Path]:
    """
    Dynamically discovers the eval queries JSONL file and SQLite database in the given directory.
    """
    # 1. Find SQLite DB
    sqlite_candidates = list(lang_dir.glob("*.sqlite")) + list(lang_dir.glob("*.db"))
    if not sqlite_candidates:
        raise FileNotFoundError(f"No SQLite database (*.sqlite or *.db) found in: {lang_dir}")
    sqlite_file = sqlite_candidates[0]

    # 2. Find Queries / Eval JSONL file
    eval_candidates = []
    for f in lang_dir.glob("*.jsonl"):
        fname = f.name.lower()
        # Filter out passages or metadata files
        if "passage" in fname or "metadata" in fname or "golden" in fname:
            continue
        eval_candidates.append(f)

    if not eval_candidates:
        # Fallback to any jsonl that is not metadata/passage
        for f in lang_dir.glob("*.jsonl"):
            if "passage" not in f.name.lower() and "metadata" not in f.name.lower():
                eval_candidates.append(f)

    if not eval_candidates:
        raise FileNotFoundError(f"No valid evaluation/query JSONL file found in: {lang_dir}")

    # Prioritize files with 'eval' or 'query' in the name
    eval_file = sorted(eval_candidates, key=lambda x: (
        0 if "eval" in x.name.lower() else (1 if "quer" in x.name.lower() else 2)
    ))[0]

    return eval_file, sqlite_file


def is_unanswerable_text(answer: str, lang_code_or_name: str) -> bool:
    """
    Checks if an answer string matches unanswerable/negative patterns in the given or generic languages.
    """
    if not answer:
        return True
    
    clean_ans = answer.strip().lower()
    if not clean_ans:
        return True

    # Generic check
    for phrase in GENERIC_UNANSWERABLE:
        if clean_ans == phrase or f"{phrase}." in clean_ans:
            return True

    # Language-specific triggers
    lang_key = lang_code_or_name.lower()
    triggers = MULTILINGUAL_UNANSWERABLE_TRIGGERS.get(lang_key, [])
    for phrase in triggers:
        if phrase.lower() in clean_ans:
            return True

    return False


def create_stratified_sample(records: List[Dict[str, Any]], target_size: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Creates a stratified sample across query_type for balanced benchmarking.
    """
    if len(records) <= target_size:
        return list(records)

    random.seed(seed)
    records_by_type = defaultdict(list)
    for r in records:
        q_type = r.get("query_type") or "UNKNOWN"
        records_by_type[q_type].append(r)

    total = len(records)
    sampled = []

    # Proportional allocation
    for q_type, type_records in records_by_type.items():
        type_target = max(1, int(round((len(type_records) / total) * target_size)))
        sample = random.sample(type_records, min(type_target, len(type_records)))
        sampled.extend(sample)

    # Adjust exact target size if rounding caused delta
    if len(sampled) > target_size:
        sampled = random.sample(sampled, target_size)
    elif len(sampled) < target_size:
        remaining = [r for r in records if r not in sampled]
        needed = target_size - len(sampled)
        if remaining:
            sampled.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(sampled)
    return sampled


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Writes a list of dictionaries to a UTF-8 JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   💾 Saved: {path.name:<32} ({len(records):,} records | {path.stat().st_size:,} bytes)")


def generate_dataset_for_language(
    lang_name: str,
    base_dir: Path,
    sample_sizes: Tuple[int, int] = (100, 500)
) -> bool:
    """
    Unified pipeline to build Golden Evaluation Datasets for any language.
    Reads from: `base_dir / voice_rag_builder / <lang_name>`
    Writes to:  `base_dir / Data / <lang_name>`
    """
    data_dir = base_dir / "voice_rag_builder" / lang_name
    output_dir = base_dir / "Data" / lang_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"🌐 UNIFIED GOLDEN DATASET BUILDER — TARGET LANGUAGE: [{lang_name.upper()}]")
    print("=" * 75)

    if not data_dir.exists():
        print(f"❌ Error: Data directory does not exist: {data_dir}")
        return False

    try:
        eval_queries_file, sqlite_file = find_language_files(data_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False

    print(f"📁 Source Query File   : {eval_queries_file.name}")
    print(f"📁 SQLite Database     : {sqlite_file.name}")
    print(f"🎯 Output Destination  : {output_dir}")

    # 1. Connect to SQLite and fetch positive ground-truth chunks (is_selected = 1)
    print("\n🔍 [1/4] Loading positive ground-truth chunks from SQLite metadata...")
    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    # Detect table and column names dynamically
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cur.fetchall()]
    table_name = "chunks" if "chunks" in tables else tables[0]

    cur.execute(f"PRAGMA table_info({table_name})")
    cols = {c[1] for c in cur.fetchall()}
    
    qid_col = "query_id" if "query_id" in cols else "qid"
    chunk_col = "chunk_id" if "chunk_id" in cols else ("id" if "id" in cols else "rowid")
    passage_col = "passage_id" if "passage_id" in cols else chunk_col
    text_col = "text" if "text" in cols else "passage_text"
    sel_col = "is_selected" if "is_selected" in cols else ("selected" if "selected" in cols else "1")
    lang_col = "language" if "language" in cols else ("lang" if "lang" in cols else "NULL")

    query_sql = f"""
        SELECT {qid_col}, {chunk_col}, {passage_col}, {text_col}, {lang_col}
        FROM {table_name}
        WHERE {sel_col} = 1
        ORDER BY {qid_col}, {chunk_col}
    """
    
    cur.execute(query_sql)
    rows = cur.fetchall()
    conn.close()

    # Detect dataset language code from actual SQLite rows
    sample_lang = None
    for _, _, _, _, l in rows:
        if l:
            sample_lang = l.strip()
            break

    cfg = resolve_language_metadata(lang_name, sample_lang)
    lang_code = cfg["code"]
    lang_display = cfg["name"]

    pos_passages_by_qid = defaultdict(list)
    for qid, chunk_id, passage_id, text, l in rows:
        pos_passages_by_qid[qid].append({
            "chunk_id": chunk_id,
            "passage_id": passage_id,
            "text": text.strip() if text else "",
            "language": l or lang_code
        })

    print(f"✅ Loaded {len(rows):,} positive chunks across {len(pos_passages_by_qid):,} unique queries.")
    print(f"ℹ️  Detected Metadata: Name='{lang_display}', Code='{lang_code}', Locale='{cfg['locale']}'")

    # 2. Parse eval queries and build ground-truth tuples
    print("\n🔍 [2/4] Filtering and assembling Golden Question-Context-Answer records...")
    raw_eval_count = 0
    unanswerable_count = 0
    no_pos_passage_count = 0

    golden_records = []
    type_counter = Counter()

    with open(eval_queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            raw_eval_count += 1
            try:
                data = json.loads(line_str)
            except Exception:
                continue

            qid = data.get("query_id") or data.get("id") or data.get("qid")
            q_text = (data.get("query") or data.get("question") or data.get("text") or "").strip()
            
            # Answer field can be 'answer', 'answers', or 'ground_truth'
            ans_raw = data.get("answer") or data.get("ground_truth") or data.get("answers") or ""
            if isinstance(ans_raw, list):
                ans_text = ans_raw[0].strip() if ans_raw else ""
            else:
                ans_text = str(ans_raw).strip()

            q_type = data.get("query_type") or data.get("type") or "UNKNOWN"

            # Filter unanswerable / missing answers
            if is_unanswerable_text(ans_text, lang_code) or is_unanswerable_text(ans_text, lang_name):
                unanswerable_count += 1
                continue

            # Must have positive passages
            if qid not in pos_passages_by_qid:
                no_pos_passage_count += 1
                continue

            pos_items = pos_passages_by_qid[qid]
            contexts = [p["text"] for p in pos_items if p["text"]]
            p_ids = [p["passage_id"] for p in pos_items]
            c_ids = [p["chunk_id"] for p in pos_items]

            if not contexts:
                continue

            record = {
                "query_id": qid,
                "query_type": q_type,
                "question": q_text,
                "ground_truth_answer": ans_text,
                "ground_truth_contexts": contexts,
                "ground_truth_passage_ids": p_ids,
                "ground_truth_chunk_ids": c_ids,
                "metadata": {
                    "language": lang_code,
                    "language_name": lang_display,
                    "num_positive_passages": len(contexts),
                    "question_char_len": len(q_text),
                    "answer_char_len": len(ans_text),
                    "total_context_char_len": sum(len(c) for c in contexts)
                }
            }
            golden_records.append(record)
            type_counter[q_type] += 1

    print(f"✅ Filtered {len(golden_records):,} valid Golden dataset records.")
    print(f"   - Raw queries analyzed        : {raw_eval_count:,}")
    print(f"   - Unanswerable excluded       : {unanswerable_count:,}")
    print(f"   - Missing pos-passage excluded: {no_pos_passage_count:,}")
    print(f"   - Query Type breakdown        : {dict(type_counter)}")

    if not golden_records:
        print(f"⚠️ Warning: No valid golden records created for {lang_name}.")
        return False

    # 3. Create Stratified Subsets
    size_fast, size_std = sample_sizes
    print(f"\n🔍 [3/4] Generating stratified benchmark samples ({size_fast} & {size_std} records)...")
    sample_fast = create_stratified_sample(golden_records, size_fast)
    sample_std = create_stratified_sample(golden_records, size_std)

    # 4. Save to Data/<lang_name>
    print("\n🔍 [4/4] Writing datasets, summary scorecards, and documentation...")
    full_path = output_dir / "golden_dataset_full.jsonl"
    sample_fast_path = output_dir / f"golden_dataset_sample_{size_fast}.jsonl"
    sample_std_path = output_dir / f"golden_dataset_sample_{size_std}.jsonl"
    summary_path = output_dir / "dataset_summary.json"
    readme_path = output_dir / "README.md"

    save_jsonl(full_path, golden_records)
    save_jsonl(sample_fast_path, sample_fast)
    save_jsonl(sample_std_path, sample_std)

    avg_q_len = round(sum(r["metadata"]["question_char_len"] for r in golden_records) / len(golden_records), 2)
    avg_a_len = round(sum(r["metadata"]["answer_char_len"] for r in golden_records) / len(golden_records), 2)
    avg_pos = round(sum(r["metadata"]["num_positive_passages"] for r in golden_records) / len(golden_records), 2)

    summary = {
        "dataset_name": f"{lang_display} Voice & Text RAG Golden Evaluation Dataset",
        "version": "1.0",
        "language": f"{lang_display} ({cfg['locale']})",
        "language_code": lang_code,
        "description": f"Curated golden question-context-answer triples for {lang_display} retrieval and generation evaluation.",
        "statistics": {
            "total_golden_records": len(golden_records),
            "raw_eval_queries_analyzed": raw_eval_count,
            "unanswerable_queries_excluded": unanswerable_count,
            "missing_positive_passage_excluded": no_pos_passage_count,
            "query_types": dict(type_counter),
            "avg_question_char_length": avg_q_len,
            "avg_answer_char_length": avg_a_len,
            "avg_positive_passages_per_query": avg_pos
        },
        "subsets": {
            "golden_dataset_full.jsonl": f"{len(golden_records)} records (Comprehensive evaluation)",
            f"golden_dataset_sample_{size_std}.jsonl": f"{len(sample_std)} records (Standard benchmark)",
            f"golden_dataset_sample_{size_fast}.jsonl": f"{len(sample_fast)} records (Fast test benchmark)"
        },
        f"sample_{size_fast}_type_breakdown": dict(Counter(r["query_type"] for r in sample_fast)),
        f"sample_{size_std}_type_breakdown": dict(Counter(r["query_type"] for r in sample_std))
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"   💾 Saved: {summary_path.name:<32}")

    # Generate Markdown documentation
    readme_content = f"""# {lang_display} Golden Evaluation Dataset for Voice & Text RAG

## Overview
- **Language**: {lang_display} (`{lang_code}`, `{cfg['locale']}`)
- **Total Valid Golden Records**: {len(golden_records):,}
- **Average Question Length**: {avg_q_len} characters
- **Average Answer Length**: {avg_a_len} characters
- **Average Positive Passages**: {avg_pos}

## Dataset Files
| Filename | Records | Purpose |
|---|---|---|
| `golden_dataset_full.jsonl` | {len(golden_records):,} | Complete golden ground-truth evaluation set |
| `golden_dataset_sample_{size_std}.jsonl` | {len(sample_std):,} | Standard evaluation benchmark |
| `golden_dataset_sample_{size_fast}.jsonl` | {len(sample_fast):,} | Rapid CI/CD test benchmark |
| `dataset_summary.json` | - | Summary statistics and type distributions |

## Query Type Breakdown
```json
{json.dumps(dict(type_counter), indent=2, ensure_ascii=False)}
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"   💾 Saved: {readme_path.name:<32}")

    print(f"\n🎉 Golden dataset generation finished successfully for {lang_display} in: {output_dir}")
    return True


def discover_available_languages(base_dir: Path) -> List[str]:
    """Finds all language subdirectories present in voice_rag_builder."""
    builder_dir = base_dir / "voice_rag_builder"
    if not builder_dir.exists():
        return []
    languages = []
    for item in builder_dir.iterdir():
        if item.is_dir():
            # Check if directory has sqlite or jsonl files
            has_db = len(list(item.glob("*.sqlite"))) > 0 or len(list(item.glob("*.db"))) > 0
            if has_db:
                languages.append(item.name)
    return sorted(languages)


def main():
    base_dir = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path("d:/Harsh/Voice RAG System/v3")
    available_langs = discover_available_languages(base_dir)

    parser = argparse.ArgumentParser(
        description="Unified Multi-Language Golden Evaluation Dataset Generator for Voice & Text RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_golden_dataset.py --lang hindi
  python generate_golden_dataset.py --lang gujarati
  python generate_golden_dataset.py --lang marathi
  python generate_golden_dataset.py --all
  python generate_golden_dataset.py --list
        """
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default=None,
        help=f"Specific language directory name to build. Discovered: {available_langs}"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Automatically generate golden datasets for all language directories found in voice_rag_builder"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered language directories in voice_rag_builder"
    )
    parser.add_argument(
        "positional_lang",
        nargs="?",
        type=str,
        default=None,
        help="Optional positional language directory name"
    )

    args = parser.parse_args()

    if args.list:
        print(f"\n🌐 Discovered language directories in 'voice_rag_builder/':")
        for idx, l in enumerate(available_langs, 1):
            cfg = resolve_language_metadata(l)
            print(f"  [{idx}] {l:<15} -> {cfg['name']} ({cfg['code']})")
        return

    target_lang = args.lang or args.positional_lang

    if args.all:
        if not available_langs:
            print(f"❌ No valid language directories found in {base_dir / 'voice_rag_builder'}")
            sys.exit(1)
        print(f"🌐 Found {len(available_langs)} languages: {available_langs}. Processing all...")
        for l in available_langs:
            generate_dataset_for_language(l, base_dir)
    elif target_lang:
        generate_dataset_for_language(target_lang, base_dir)
    else:
        # If no arguments provided, automatically list and process all available languages
        print("💡 No specific language provided. Discovered languages in voice_rag_builder:")
        for idx, l in enumerate(available_langs, 1):
            cfg = resolve_language_metadata(l)
            print(f"  [{idx}] {l:<15} ({cfg['name']})")
        
        if available_langs:
            print(f"\n🚀 Running dataset generator for all {len(available_langs)} languages...")
            for l in available_langs:
                generate_dataset_for_language(l, base_dir)
        else:
            print("❌ No language directories found in voice_rag_builder.")


if __name__ == "__main__":
    main()

