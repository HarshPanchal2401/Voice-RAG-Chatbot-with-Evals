"""
Multilingual RAG Re-ranker Module
=================================
One class, `RAGReranker`, with three backends (RAG_RERANKER_BACKEND):

- ``llm`` (default): ONE Groq chat call with a short timeout and no retries grades every
  candidate 0-3 (JSON). ``rerank_score = grade / 3``. On any failure it falls back to lexical.
- ``cross_encoder`` (opt-in): local transformers sequence-classification model
  (RAG_CROSS_ENCODER_MODEL, default BAAI/bge-reranker-v2-m3), lazily loaded once per process,
  batched (query, passage) pairs, sigmoid scores; fp16 on CUDA, optional int8 on CPU.
- ``lexical``: whole-token overlap that works for Gujarati / Hindi (whitespace tokens with
  Unicode punctuation stripped; Python's ``\\w`` would split words at vowel signs / virama).

Documents scoring below the minimum score are dropped; if every candidate is dropped an
empty list is returned and the pipeline answers with the no-answer sentence.
"""

import json
import os
import re
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.config import (
    RERANKER_BACKEND,
    RERANKER_BACKENDS,
    RERANKER_MODEL_NAME,
    DEFAULT_JUDGE_MODEL,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_QUANTIZE,
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_MAX_LENGTH,
    RERANK_MIN_SCORE,
    RERANK_DEFAULT_MIN_SCORES,
    RERANK_TIMEOUT,
    RERANK_MAX_TOKENS,
    RERANK_PASSAGE_CHARS,
    RERANK_REASONING_EFFORT,
    DEVICE,
)
from core.tracing import traceable, add_run_metadata, get_fast_llm_client, GROQ_BASE_URL
from pipeline.generator import create_chat_completion, strip_think

MAX_GRADE = 3


# ------------------------------------------------------------
# Lexical scoring
# ------------------------------------------------------------

_STOPWORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "for", "and", "or",
    "what", "who", "whom", "when", "where", "why", "how", "which", "do", "does", "did", "it",
    "this", "that", "by", "with", "as", "at", "be", "can", "from",
    # Hindi
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "से", "को", "और", "या", "क्या",
    "कौन", "कब", "कहाँ", "कहां", "कैसे", "कितना", "कितने", "कितनी", "यह", "वह", "एक", "पर",
    "भी", "हो", "ने", "तो", "ही",
    # Gujarati
    "છે", "છો", "હતો", "હતી", "હતું", "હતા", "નો", "ની", "નું", "ના", "માં", "થી", "ને", "અને",
    "કે", "શું", "કોણ", "ક્યારે", "ક્યાં", "કેવી", "કેવી", "કેટલા", "કેટલી", "કેટલું", "આ", "તે",
    "એક", "પર", "પણ", "હોય", "જ",
}
_STOPWORD_WEIGHT = 0.2


def lexical_tokens(text: str) -> List[str]:
    """Whitespace tokens with Unicode punctuation stripped (keeps Indic vowel signs / virama)."""
    text = unicodedata.normalize("NFC", text or "").lower()
    tokens = []
    for raw in text.split():
        tok = "".join(ch for ch in raw if not unicodedata.category(ch).startswith("P"))
        if tok:
            tokens.append(tok)
    return tokens


def _token_weight(tok: str) -> float:
    if tok in _STOPWORDS:
        return _STOPWORD_WEIGHT
    return float(min(len(tok), 8)) / 8.0 + 0.5


def lexical_score(query: str, text: str, query_tokens: Optional[Sequence[str]] = None) -> float:
    """
    Weighted share of query tokens found in the passage, in [0, 1].
    Exact whole-token match = full credit; a passage token that starts with a query token of
    >= 3 characters (inflected form, e.g. ભારત -> ભારતની) = half credit. No substring matching.
    """
    q = list(dict.fromkeys(query_tokens if query_tokens is not None else lexical_tokens(query)))
    if not q:
        return 0.0
    doc_tokens = set(lexical_tokens(text))
    if not doc_tokens:
        return 0.0
    total = matched = 0.0
    for tok in q:
        w = _token_weight(tok)
        total += w
        if tok in doc_tokens:
            matched += w
        elif len(tok) >= 3 and any(d.startswith(tok) for d in doc_tokens):
            matched += 0.5 * w
    return round(matched / total, 4) if total else 0.0


# ------------------------------------------------------------
# LLM response parsing
# ------------------------------------------------------------

def _coerce_index(value: Any, n: int) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*\[?\s*(?:passage\s*)?(\d+)\s*\]?\s*", value, re.IGNORECASE)
        value = int(m.group(1)) if m else None
    if isinstance(value, int) and 0 <= value < n:
        return value
    return None


def _coerce_grade(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        g = float(value)
    except (TypeError, ValueError):
        return None
    if g != g:  # NaN
        return None
    return max(0.0, min(float(MAX_GRADE), g))


def _extract_json(raw: str) -> Dict[str, Any]:
    raw = strip_think(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()
    elif "{" in raw and "}" in raw:
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("reranker response is not a JSON object")
    return parsed


def parse_llm_rerank(raw: str, n: int) -> Tuple[Dict[int, float], List[int], str]:
    """
    Parses the grader output into ({index: grade 0..3}, ranking without duplicates, reason).
    Accepts grades as {"0": 3, ...}, [3, 0, ...] (positional) or [{"id": 0, "grade": 3}, ...];
    ignores out-of-range / non-integer / duplicate indices.
    """
    parsed = _extract_json(raw)
    grades: Dict[int, float] = {}
    raw_grades = parsed.get("grades", parsed.get("scores"))
    if isinstance(raw_grades, dict):
        for k, v in raw_grades.items():
            idx, g = _coerce_index(k, n), _coerce_grade(v)
            if idx is not None and g is not None and idx not in grades:
                grades[idx] = g
    elif isinstance(raw_grades, list):
        for pos, item in enumerate(raw_grades):
            if isinstance(item, dict):
                idx = _coerce_index(item.get("id", item.get("index", item.get("passage"))), n)
                g = _coerce_grade(item.get("grade", item.get("score", item.get("relevance"))))
            else:
                idx, g = (pos if pos < n else None), _coerce_grade(item)
            if idx is not None and g is not None and idx not in grades:
                grades[idx] = g

    ranking: List[int] = []
    for v in parsed.get("ranking", parsed.get("ranked_indices", [])) or []:
        idx = _coerce_index(v, n)
        if idx is not None and idx not in ranking:
            ranking.append(idx)

    reason = parsed.get("reason", parsed.get("top_passage_reason", "")) or ""
    return grades, ranking, str(reason)[:300]


_SYSTEM_PROMPT = (
    "You are a precise multilingual search relevance grader. Passages and questions may be in "
    "Gujarati, Hindi or English. Output only valid JSON."
)


def build_llm_rerank_prompt(query: str, texts: Sequence[str], max_chars: int = RERANK_PASSAGE_CHARS) -> str:
    lines = [f"Question: {query}", "", "Passages:"]
    for i, text in enumerate(texts):
        snippet = re.sub(r"\s+", " ", text or "").strip()[:max_chars]
        lines.append(f"[{i}] {snippet}")
    ids = ", ".join(f'"{i}": <0-3>' for i in range(len(texts)))
    lines += [
        "",
        "Grade how well each passage answers the question:",
        "3 = directly answers the question",
        "2 = contains part of the answer or key supporting facts",
        "1 = same topic but does not answer the question",
        "0 = unrelated",
        "",
        "Return only JSON of this shape:",
        f'{{"grades": {{{ids}}}, "ranking": [passage ids, most relevant first], '
        '"reason": "<one short sentence on the best passage>"}',
        "Grade every passage exactly once.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------
# Cross-encoder (process-wide lazy singleton)
# ------------------------------------------------------------

_CE_LOCK = threading.Lock()
_CE_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_cross_encoder(model_name: str, quantize: bool = CROSS_ENCODER_QUANTIZE) -> Dict[str, Any]:
    """Loads (once per process) tokenizer + sequence-classification model. Thread-safe."""
    with _CE_LOCK:
        entry = _CE_CACHE.get(model_name)
        if entry is not None:
            return entry
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        hf_token = os.environ.get("HF_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
        model.eval()
        device = torch.device(DEVICE)
        if device.type == "cuda":
            model = model.half().to(device)
        elif quantize:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        entry = {"tokenizer": tokenizer, "model": model, "device": device, "lock": threading.Lock()}
        _CE_CACHE[model_name] = entry
        print(f"✅ [reranker] Cross-encoder loaded: {model_name} ({device}{', int8' if quantize and device.type != 'cuda' else ''})")
        return entry


def register_cross_encoder(model_name: str, tokenizer: Any, model: Any, device: str = "cpu") -> None:
    """Injects an already-built tokenizer/model (tests, custom loading)."""
    import torch
    with _CE_LOCK:
        _CE_CACHE[model_name] = {
            "tokenizer": tokenizer, "model": model, "device": torch.device(device), "lock": threading.Lock(),
        }


# ------------------------------------------------------------
# Reranker
# ------------------------------------------------------------

class RAGReranker:
    """
    Multilingual re-ranker shared by all language pipelines (stateless per call, thread-safe).

    `rerank(query, documents, top_k)` -> (documents, rerank_ms). Returned docs are copies ordered
    by `rerank_score` (higher = more relevant) with `rank`, `rerank_rank`, `original_rank` and
    `rerank_reason` set; docs below the minimum score are dropped (possibly all of them).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        *,
        backend: Optional[str] = None,
        client: Any = None,
        min_score: Optional[float] = None,
        cross_encoder_model: Optional[str] = None,
        max_passage_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        backend = (backend or RERANKER_BACKEND).strip().lower().replace("-", "_")
        if backend not in RERANKER_BACKENDS:
            raise ValueError(f"Unknown reranker backend {backend!r}; expected one of {RERANKER_BACKENDS}")
        self.backend = backend
        self.model_name = model_name or RERANKER_MODEL_NAME
        self.timeout = float(timeout or RERANK_TIMEOUT)
        self.max_passage_chars = int(max_passage_chars or RERANK_PASSAGE_CHARS)
        self.max_tokens = int(max_tokens or RERANK_MAX_TOKENS)
        self.reasoning_effort = reasoning_effort or RERANK_REASONING_EFFORT
        self.cross_encoder_model = cross_encoder_model or CROSS_ENCODER_MODEL_NAME
        if min_score is None:
            min_score = RERANK_MIN_SCORE if RERANK_MIN_SCORE is not None else RERANK_DEFAULT_MIN_SCORES[backend]
        self.min_score = float(min_score)
        self._client = client
        self._api_key = (api_key or "").strip().strip("\"' ") or None

        if self.backend == "llm" and self.model_name == DEFAULT_JUDGE_MODEL:
            print(f"⚠️ [reranker] Reranker model {self.model_name} equals the judge model; "
                  "evaluation scores will be biased. Set RAG_RERANKER_MODEL to a different model.")

    # -- clients -----------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            env_key = os.environ.get("GROQ_API_KEY", "").strip().strip("\"' ")
            if self._api_key and self._api_key != env_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key, base_url=GROQ_BASE_URL, max_retries=0)
            elif env_key:
                self._client = get_fast_llm_client()
        return self._client

    def _min_score_for(self, backend_used: str) -> float:
        return self.min_score if backend_used == self.backend else RERANK_DEFAULT_MIN_SCORES[backend_used]

    # -- backends ----------------------------------------------------------

    def _lexical_scores(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        q_tokens = lexical_tokens(query)
        return [lexical_score(query, d.get("text", ""), q_tokens) for d in docs]

    def _llm_scores(self, query: str, docs: List[Dict[str, Any]]) -> Tuple[List[Optional[float]], List[int], str]:
        client = self.client
        if client is None:
            raise RuntimeError("no Groq API key configured")
        prompt = build_llm_rerank_prompt(query, [d.get("text", "") for d in docs], self.max_passage_chars)
        resp = create_chat_completion(
            client,
            model=self.model_name,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            reasoning_effort=self.reasoning_effort,
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            timeout=self.timeout,
        )
        grades, ranking, reason = parse_llm_rerank(resp.choices[0].message.content or "", len(docs))
        if not grades:
            raise ValueError("reranker returned no usable grades")
        scores = [round(grades[i] / MAX_GRADE, 4) if i in grades else None for i in range(len(docs))]
        return scores, ranking, reason

    def _cross_encoder_scores(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        import torch
        entry = _load_cross_encoder(self.cross_encoder_model)
        tokenizer, model, device = entry["tokenizer"], entry["model"], entry["device"]
        texts = [d.get("text", "") or "" for d in docs]
        scores: List[float] = []
        with entry["lock"]:
            for start in range(0, len(texts), CROSS_ENCODER_BATCH_SIZE):
                batch = texts[start:start + CROSS_ENCODER_BATCH_SIZE]
                enc = tokenizer(
                    [query] * len(batch), batch,
                    padding=True, truncation=True, max_length=CROSS_ENCODER_MAX_LENGTH, return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = model(**enc).logits.float()
                if logits.dim() == 2 and logits.shape[-1] == 2:
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.reshape(len(batch), -1)[:, 0])
                scores.extend(round(float(p), 6) for p in probs.cpu())
        return scores

    # -- main entry --------------------------------------------------------

    @staticmethod
    def _dedupe(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for d in documents:
            key = d.get("vector_id")
            if key is None:
                key = ("doc_key", d.get("doc_key")) if d.get("doc_key") else ("text", d.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        return unique

    @traceable(
        run_type="tool",
        name="rerank",
        process_inputs=lambda i: {"query": i.get("query"), "candidates": len(i.get("documents") or []), "top_k": i.get("top_k")},
        process_outputs=lambda o: {"kept": len(o[0]), "rerank_ms": o[1]} if isinstance(o, tuple) else o,
    )
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Re-ranks candidate documents for `query`. Returns (reranked_documents, rerank_ms)."""
        t0 = time.perf_counter()
        docs = self._dedupe(list(documents or []))
        if not docs:
            return [], 0.0

        backend_used = self.backend
        ranking: List[int] = []
        llm_reason = ""
        fallback_note = ""
        try:
            if self.backend == "llm":
                scores, ranking, llm_reason = self._llm_scores(query, docs)
            elif self.backend == "cross_encoder":
                scores = self._cross_encoder_scores(query, docs)
            else:
                scores = self._lexical_scores(query, docs)
        except Exception as e:
            backend_used = "lexical"
            fallback_note = f"{self.backend} failed ({type(e).__name__}: {str(e)[:160]})"
            print(f"⚠️ [reranker] {fallback_note}; using lexical fallback.")
            scores = self._lexical_scores(query, docs)
            ranking = []

        rank_pos = {idx: pos for pos, idx in enumerate(ranking)}
        order = sorted(
            range(len(docs)),
            key=lambda i: (
                scores[i] is not None,
                scores[i] if scores[i] is not None else 0.0,
                -rank_pos.get(i, len(docs)),
                -i,
            ),
            reverse=True,
        )

        min_score = self._min_score_for(backend_used)
        label = backend_used if backend_used != "llm" else f"llm:{self.model_name}"
        if fallback_note:
            label = f"lexical fallback: {fallback_note}"

        kept: List[Dict[str, Any]] = []
        dropped = 0
        for i in order:
            score = scores[i]
            if score is None or score < min_score:
                dropped += 1
                continue
            if len(kept) >= top_k:
                continue
            doc = dict(docs[i])
            new_rank = len(kept) + 1
            doc["original_rank"] = docs[i].get("original_rank") or docs[i].get("rank") or (i + 1)
            doc["rank"] = new_rank
            doc["rerank_rank"] = new_rank
            doc["rerank_score"] = float(score)
            reason = label
            if backend_used == "llm":
                reason = f"{label} grade={int(round(score * MAX_GRADE))}/{MAX_GRADE}"
                if new_rank == 1 and llm_reason:
                    reason = f"{reason}: {llm_reason}"
            doc["rerank_reason"] = reason
            kept.append(doc)

        rerank_ms = (time.perf_counter() - t0) * 1000
        add_run_metadata(
            rerank_backend=backend_used,
            rerank_model=self.model_name if backend_used == "llm" else (self.cross_encoder_model if backend_used == "cross_encoder" else None),
            rerank_candidates=len(docs),
            rerank_dropped=dropped,
            rerank_min_score=min_score,
            rerank_fallback=fallback_note or None,
        )
        return kept, rerank_ms
