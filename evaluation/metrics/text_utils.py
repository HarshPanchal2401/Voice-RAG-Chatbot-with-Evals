"""
Shared Unicode-aware text utilities for evaluation
==================================================
ONE tokenizer for every lexical metric (Gujarati, Hindi, English), used by the
offline harness, the online evaluator and the LangSmith evaluators.

Why not ``re.findall(r"\\w+")``: Python's ``\\w`` does not match Indic vowel signs
or the virama (Unicode categories Mn/Mc), so "भारत की राजधानी" is shredded into
['भ', 'रत', 'क', 'र', 'जध', 'न']. ``\\b`` has the same problem, which is why regex
word-boundary matching of Indic words silently fails.

Rules (documented, deterministic):
1. NFC-normalise the text (so composed / decomposed forms compare equal).
2. Lowercase (only affects cased scripts such as Latin; Indic scripts are uncased).
3. Remove zero-width / format characters (category ``Cf``: ZWJ, ZWNJ, BOM, ...).
4. Replace every character whose Unicode category starts with ``P`` (punctuation,
   incl. the danda ``।`` and double danda ``॥``) or ``S`` (symbols) with a space.
5. Split on whitespace. Tokens are whole words; matching is always whole-token
   (never substring), so 1-2 character fragments cannot produce spurious hits.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "normalize_text",
    "normalize_question",
    "tokenize",
    "token_set",
    "token_f1",
    "token_jaccard",
    "token_coverage",
    "contains_phrase",
    "contains_any_phrase",
    "REFUSAL_PHRASES",
    "is_refusal",
    "TOXIC_LEXICON",
    "lexical_toxicity_hits",
]


def _clean_char(ch: str) -> str:
    cat = unicodedata.category(ch)
    if cat == "Cf":
        return ""
    if cat[0] in ("P", "S"):
        return " "
    return ch


def normalize_text(text: Optional[str]) -> str:
    """NFC + lowercase + punctuation/symbols -> space + collapsed whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    cleaned = "".join(_clean_char(ch) for ch in text)
    return " ".join(cleaned.split())


def normalize_question(text: Optional[str]) -> str:
    """
    Key used for *exact* golden-question lookup: NFC, lowercase, whitespace collapsed,
    trailing sentence punctuation (? । . ! ॥) removed. Inner punctuation is kept, so
    this is an exact match up to case/whitespace/final punctuation - never fuzzy.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = " ".join(t.split())
    return t.rstrip("?？.!।॥ ").strip()


def tokenize(text: Optional[str]) -> List[str]:
    """Unicode-aware whitespace tokenizer (see module docstring)."""
    return normalize_text(text).split()


def token_set(text: Optional[str]) -> set:
    return set(tokenize(text))


def _tokens_of(x: Iterable[str] | str | None) -> set:
    if not x:
        return set()
    if isinstance(x, str):
        return token_set(x)
    out: set = set()
    for item in x:
        out |= token_set(item)
    return out


def token_f1(prediction: Optional[str], reference: Optional[str]) -> Dict[str, Optional[float]]:
    """SQuAD-style multiset token precision / recall / F1 on the shared tokenizer.

    Returns None values when either side has no tokens (undefined, not 0)."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return {"precision": None, "recall": None, "f1": None}
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = same / len(pred)
    r = same / len(ref)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4)}


def token_jaccard(a: Iterable[str] | str | None, b: Iterable[str] | str | None) -> Optional[float]:
    """Jaccard overlap of token sets. Accepts raw strings or iterables of strings."""
    sa, sb = _tokens_of(a), _tokens_of(b)
    if not sa or not sb:
        return None
    return round(len(sa & sb) / len(sa | sb), 4)


def token_coverage(reference: Optional[str], texts: Sequence[str] | str | None) -> Optional[float]:
    """Fraction of the reference's distinct tokens that appear (whole-token) in `texts`."""
    ref = token_set(reference)
    if not ref:
        return None
    pool = _tokens_of(texts)
    return round(len(ref & pool) / len(ref), 4)


def contains_phrase(text_tokens: Sequence[str], phrase: str) -> bool:
    """True when the tokenized `phrase` occurs as a contiguous whole-token run."""
    ptoks = tokenize(phrase)
    n = len(ptoks)
    if n == 0 or n > len(text_tokens):
        return False
    toks = list(text_tokens)
    first = ptoks[0]
    for i in range(len(toks) - n + 1):
        if toks[i] == first and toks[i:i + n] == ptoks:
            return True
    return False


def contains_any_phrase(text: Optional[str], phrases: Iterable[str]) -> List[str]:
    toks = tokenize(text)
    return [p for p in phrases if contains_phrase(toks, p)]


# ------------------------------------------------------------------
# Refusal detection (single shared list - used offline AND online)
# ------------------------------------------------------------------
# Phrases come from the production system prompts (core.config.SYSTEM_PROMPTS) plus
# common paraphrases. Matched as whole-token sequences after normalisation.
REFUSAL_PHRASES: Dict[str, List[str]] = {
    "gu": [
        "પૂરતી માહિતી ઉપલબ્ધ નથી",
        "માહિતી ઉપલબ્ધ નથી",
        "પર્યાપ્ત માહિતી ઉપલબ્ધ નથી",
        "પર્યાપ્ત જાણકારી ઉપલબ્ધ નથી",
        "જાણકારી ઉપલબ્ધ નથી",
        "મને ખબર નથી",
    ],
    "hi": [
        "पर्याप्त जानकारी उपलब्ध नहीं",
        "जानकारी उपलब्ध नहीं",
        "जानकारी नहीं दी गई",
        "मुझे नहीं पता",
        "मुझे जानकारी नहीं",
    ],
    "en": [
        "not enough information",
        "insufficient information",
        "does not contain enough information",
        "no information available",
        "i don't know",
        "i do not know",
    ],
}


def is_refusal(answer: Optional[str], no_answer_flag: Optional[bool] = None) -> bool:
    """
    True when the answer declines to answer. The pipeline's own ``no_answer`` flag wins
    when provided; otherwise whole-token phrase matching over all languages is used.
    An empty answer counts as a refusal.
    """
    if no_answer_flag is not None:
        return bool(no_answer_flag)
    if not answer or not str(answer).strip():
        return True
    toks = tokenize(answer)
    for phrases in REFUSAL_PHRASES.values():
        for p in phrases:
            if contains_phrase(toks, p):
                return True
    return False


# ------------------------------------------------------------------
# Lexical toxicity lexicon (heuristic gate, reported separately - never a judge substitute)
# ------------------------------------------------------------------
TOXIC_LEXICON: Dict[str, List[str]] = {
    "en": ["bastard", "bitch", "idiot", "stupid", "moron", "kill yourself", "retard"],
    "gu": ["ગાંડો", "ગાંડી", "મૂર્ખ", "હરામી", "બકવાસ", "નાલાયક"],
    "hi": ["कमीना", "कमीने", "कुत्ता", "कुत्ते", "मूर्ख", "बकवास", "हरामी", "बेवकूफ"],
}


def lexical_toxicity_hits(text: Optional[str]) -> List[str]:
    """Whole-token matches of the toxicity lexicon (any language)."""
    toks = tokenize(text)
    hits: List[str] = []
    for words in TOXIC_LEXICON.values():
        for w in words:
            if contains_phrase(toks, w):
                hits.append(w)
    return hits
