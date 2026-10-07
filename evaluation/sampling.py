"""
Stratified, grouped sampling of golden questions
================================================
* ``stratified_group_sample`` - picks ``n`` distinct groups (query_ids) stratified by
  ``query_type`` with largest-remainder proportional allocation (+ optional minimum per
  stratum). All records of a chosen group are kept together, so the Gujarati and Hindi
  versions of the same question are always paired. Deterministic for a given seed and
  record set (input order does not matter).
* ``is_context_dependent`` - documented filter for questions that need their passage to
  make sense ("Is she married?"): a question with <= 5 tokens that contains a
  personal pronoun (he/she/they/his/her/its... in Gujarati, Hindi or English).
* ``ambiguous_questions`` - normalised question texts that occur under more than one
  query_id (cannot be matched to a single gold record).

Used by ``evaluation.combine_datasets eval-sets`` and by the data loader for ``--sample N``.
"""

from __future__ import annotations

import random
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from evaluation.metrics.text_utils import normalize_question, tokenize

CONTEXT_DEPENDENT_MAX_TOKENS = 5

PERSONAL_PRONOUNS: Dict[str, Set[str]] = {
    "en": set("he she they him her his hers its their theirs them it".split()),
    "hi": set("वह वे वो उसका उसकी उसके उसने उसे उन्होंने उनका उनकी उनके उन्हें".split()),
    "gu": set(
        "તે તેઓ તેનું તેની તેના તેને તેણે તેમણે તેમનું તેમની તેમના તેમને તેણી તેણીએ તેણીનું "
        "તેણીની તેણીના એનું એની એના એણે".split()
    ),
}
ALL_PRONOUNS: Set[str] = set().union(*PERSONAL_PRONOUNS.values())


def is_context_dependent(question: str, max_tokens: int = CONTEXT_DEPENDENT_MAX_TOKENS) -> bool:
    toks = tokenize(question)
    if not toks:
        return True
    return len(toks) <= max_tokens and any(t in ALL_PRONOUNS for t in toks)


def ambiguous_questions(records: Iterable[Dict[str, Any]]) -> Set[str]:
    by_q: Dict[str, Set[Any]] = {}
    for r in records:
        key = normalize_question(r.get("question", ""))
        by_q.setdefault(key, set()).add(r.get("query_id"))
    return {q for q, ids in by_q.items() if len(ids) > 1}


def _sort_key(x: Any) -> Tuple[int, Any]:
    return (0, x) if isinstance(x, (int, float)) else (1, str(x))


def allocate(stratum_sizes: Dict[str, int], n: int, min_per_stratum: int = 0) -> Dict[str, int]:
    """Largest-remainder proportional allocation with an optional floor per stratum."""
    total = sum(stratum_sizes.values())
    if n > total:
        raise ValueError(f"requested {n} groups but only {total} are eligible")
    if total == 0 or n == 0:
        return {k: 0 for k in stratum_sizes}
    exact = {k: n * v / total for k, v in stratum_sizes.items()}
    alloc = {k: int(exact[k]) for k in stratum_sizes}
    rem = n - sum(alloc.values())
    for k in sorted(stratum_sizes, key=lambda s: (-(exact[s] - alloc[s]), s))[:rem]:
        alloc[k] += 1
    if min_per_stratum > 0:
        for k, size in stratum_sizes.items():
            floor = min(min_per_stratum, size)
            alloc[k] = max(alloc[k], floor)
        # Remove the surplus from the strata that are furthest above their floor.
        surplus = sum(alloc.values()) - n
        while surplus > 0:
            k = max(sorted(alloc), key=lambda s: alloc[s] - min(min_per_stratum, stratum_sizes[s]))
            if alloc[k] <= min(min_per_stratum, stratum_sizes[k]):
                raise ValueError("min_per_stratum too large for n")
            alloc[k] -= 1
            surplus -= 1
    return alloc


def stratified_group_sample(
    records: Sequence[Dict[str, Any]],
    n_groups: int,
    seed: int = 42,
    strata_key: str = "query_type",
    group_key: str = "query_id",
    min_per_stratum: int = 0,
) -> Tuple[List[Any], Dict[str, int]]:
    """Returns (selected group ids in a deterministic shuffled order, allocation per stratum)."""
    group_stratum: Dict[Any, str] = {}
    for r in records:
        g = r.get(group_key)
        s = str(r.get(strata_key) or "UNKNOWN")
        if g in group_stratum and group_stratum[g] != s:
            raise ValueError(f"group {g!r} has inconsistent {strata_key}: {group_stratum[g]} vs {s}")
        group_stratum[g] = s
    strata: Dict[str, List[Any]] = OrderedDict()
    for g in sorted(group_stratum, key=_sort_key):
        strata.setdefault(group_stratum[g], []).append(g)
    strata = OrderedDict(sorted(strata.items()))
    alloc = allocate({k: len(v) for k, v in strata.items()}, n_groups, min_per_stratum)
    rng = random.Random(seed)
    chosen: List[Any] = []
    for s, ids in strata.items():
        chosen.extend(rng.sample(ids, alloc[s]))
    rng.shuffle(chosen)
    return chosen, alloc


def select_records(records: Sequence[Dict[str, Any]], group_ids: Sequence[Any], group_key: str = "query_id",
                   language_order: Sequence[str] = ("gu", "hi")) -> List[Dict[str, Any]]:
    """Records of the chosen groups, in the order of `group_ids` (languages paired)."""
    by_group: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r.get(group_key), []).append(r)
    order = {lang: i for i, lang in enumerate(language_order)}
    out: List[Dict[str, Any]] = []
    for g in group_ids:
        out.extend(sorted(by_group.get(g, []), key=lambda r: order.get(r.get("language"), 99)))
    return out


def eligibility(records_by_lang: Dict[str, List[Dict[str, Any]]], drop_context_dependent: bool = True
                ) -> Tuple[Set[Any], Counter]:
    """query_ids usable in every language + counts of exclusion reasons."""
    reasons: Counter = Counter()
    ids_per_lang = []
    bad: Set[Any] = set()
    for lang, recs in records_by_lang.items():
        amb = ambiguous_questions(recs)
        ids = set()
        for r in recs:
            qid = r.get("query_id")
            ids.add(qid)
            if not (r.get("ground_truth_answer") or "").strip():
                reasons[f"{lang}:empty_reference_answer"] += 1
                bad.add(qid)
            elif not r.get("ground_truth_passage_ids"):
                reasons[f"{lang}:no_gold_passages"] += 1
                bad.add(qid)
            elif normalize_question(r.get("question", "")) in amb:
                reasons[f"{lang}:ambiguous_duplicate_question"] += 1
                bad.add(qid)
            elif drop_context_dependent and is_context_dependent(r.get("question", "")):
                reasons[f"{lang}:context_dependent_question"] += 1
                bad.add(qid)
        ids_per_lang.append(ids)
    common = set.intersection(*ids_per_lang) if ids_per_lang else set()
    reasons["not_in_all_languages"] = len(set.union(*ids_per_lang) - common) if ids_per_lang else 0
    return common - bad, reasons


def sample_eval_records(records_by_lang: Dict[str, List[Dict[str, Any]]], n: int, seed: int = 42,
                        drop_context_dependent: bool = True, min_per_stratum: int = 0) -> Dict[str, Any]:
    """Stratified, language-paired sample of `n` distinct questions (query_ids)."""
    eligible, reasons = eligibility(records_by_lang, drop_context_dependent)
    first_lang = next(iter(records_by_lang))
    pool = [r for r in records_by_lang[first_lang] if r.get("query_id") in eligible]
    ids, alloc = stratified_group_sample(pool, n, seed=seed, min_per_stratum=min_per_stratum)
    selected = {lang: select_records(recs, ids) for lang, recs in records_by_lang.items()}
    return {"query_ids": ids, "allocation": alloc, "records": selected,
            "eligible": len(eligible), "exclusions": dict(reasons)}


def stratified_sample_records(records: Sequence[Dict[str, Any]], n: int, seed: int = 42,
                              min_per_stratum: int = 0) -> List[Dict[str, Any]]:
    """Convenience for one language or an already-combined list (groups by query_id)."""
    ids, _ = stratified_group_sample(records, n, seed=seed, min_per_stratum=min_per_stratum)
    return select_records(records, ids)


def language_of(record: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
    lang = record.get("language") or (record.get("metadata") or {}).get("language") or default
    return lang
