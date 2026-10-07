"""
Multilingual Prompting & Generation Layer
=========================================
Centralized system prompts, context construction, conversation memory, query condensing,
answer-language detection, sentence splitting (TTS) and Groq LLM streaming.
"""

import re
import threading
import time
from typing import Any, Dict, Generator, Iterable, List, Optional

from core.config import (
    GROQ_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPTS,
    LANGUAGE_METADATA,
    ANSWER_LANGUAGE_META,
    NO_ANSWER_MESSAGES,
    HISTORY_TURNS,
    HISTORY_MAX_CHARS,
    CONDENSE_MODEL_NAME,
    CONDENSE_MAX_TOKENS,
    CONDENSE_TIMEOUT,
    LLM_REASONING_EFFORT,
    LLM_REASONING_FORMAT,
)
from core.tracing import get_llm_client, get_fast_llm_client, traceable

__all__ = [
    "GROQ_MODEL_NAME",
    "SYSTEM_PROMPTS",
    "normalize_lang_code",
    "normalize_answer_lang",
    "detect_script_language",
    "detect_answer_language",
    "split_sentences",
    "strip_citations",
    "strip_think",
    "is_no_answer",
    "no_answer_message",
    "normalize_history",
    "build_context",
    "build_user_prompt",
    "build_messages",
    "condense_query",
    "create_chat_completion",
    "reasoning_extra_body",
    "stream_answer",
    "generate_answer",
    "warmup_llm",
]


# ------------------------------------------------------------
# Language helpers
# ------------------------------------------------------------

def normalize_lang_code(lang: Optional[str]) -> str:
    """Normalizes any language string/alias to a canonical index language code ('gu' or 'hi')."""
    if not lang:
        return "gu"
    clean = lang.strip().lower().replace("_", "-")
    for code, meta in LANGUAGE_METADATA.items():
        if clean == code or clean in meta["aliases"] or clean.replace("-", "_") in meta["aliases"]:
            return code
    for code, meta in LANGUAGE_METADATA.items():
        if code in clean or meta["name"].lower() in clean:
            return code
    return "gu"


def normalize_answer_lang(lang: Optional[str]) -> str:
    """Canonical answer language: 'gu', 'hi' or 'en'."""
    clean = (lang or "").strip().lower().replace("_", "-")
    if clean in ("en", "eng", "english") or clean.startswith("en-"):
        return "en"
    return normalize_lang_code(clean or None)


def _script_counts(text: str):
    gu = hi = latin = 0
    for ch in text or "":
        if "઀" <= ch <= "૿":
            gu += 1
        elif "ऀ" <= ch <= "ॿ":
            hi += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return gu, hi, latin


def detect_script_language(text: str) -> Optional[str]:
    """
    Detects language based on Unicode script distribution:
    - Gujarati: U+0A80 to U+0AFF
    - Hindi (Devanagari): U+0900 to U+097F
    Returns 'gu', 'hi', or None if no Indic characters detected.
    """
    if not text:
        return None
    gu_count, hi_count, _ = _script_counts(text)
    if gu_count > hi_count and gu_count > 0:
        return "gu"
    if hi_count > gu_count and hi_count > 0:
        return "hi"
    return None


def detect_answer_language(text: str, index_lang: str = "gu") -> str:
    """
    Language the answer should be written in:
    'en' when the text has Latin letters and no Gujarati/Devanagari characters, otherwise the
    dominant Indic script ('gu' / 'hi'), falling back to the index language.
    """
    gu, hi, latin = _script_counts(text or "")
    if gu == 0 and hi == 0:
        return "en" if latin > 0 else normalize_lang_code(index_lang)
    return "gu" if gu >= hi else "hi"


def no_answer_message(lang: Optional[str]) -> str:
    return NO_ANSWER_MESSAGES[normalize_answer_lang(lang)]


# ------------------------------------------------------------
# Text post-processing (citations, <think>, sentences)
# ------------------------------------------------------------

_DIGITS = r"0-9०-९૦-૯"
_LABELS = r"સંદર્ભ|संदर्भ|Source|source|SOURCE|Context|context"
_CITATION_RE = re.compile(
    rf"\s*\[\s*(?:{_LABELS})\s*[{_DIGITS}]+(?:\s*(?:,|-|–|and|और|અને)\s*(?:(?:{_LABELS})\s*)?[{_DIGITS}]+)*\s*\]"
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_citations(text: str) -> str:
    """Removes inline citation markers such as [સંદર્ભ 1], [संदर्भ 2, 3] or [Source 1]."""
    if not text:
        return ""
    out = _CITATION_RE.sub("", text)
    return re.sub(r"\s+([.,!?।॥])", r"\1", out)


def strip_think(text: str) -> str:
    """Removes hidden-reasoning blocks (<think>...</think>, unterminated or orphaned tags)."""
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub("", text)
    low = out.lower()
    if "<think>" in low:  # unterminated block: everything after the tag is reasoning
        out = out[:low.index("<think>")]
        low = out.lower()
    if "</think>" in low:  # opening tag omitted by the model: drop the reasoning prefix
        out = out[low.rindex("</think>") + len("</think>"):]
    return out


def _partial_tag_suffix(s: str, tag: str) -> int:
    """Length of the longest suffix of `s` that is a proper prefix of `tag`."""
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


class ThinkStreamFilter:
    """Incrementally removes <think>...</think> spans from streamed deltas (tags may be split across chunks)."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: List[str] = []
        while True:
            if self._in_think:
                i = self._buf.find(self.CLOSE)
                if i < 0:
                    keep = _partial_tag_suffix(self._buf, self.CLOSE)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[i + len(self.CLOSE):]
                self._in_think = False
            else:
                i = self._buf.find(self.OPEN)
                if i < 0:
                    keep = _partial_tag_suffix(self._buf, self.OPEN)
                    out.append(self._buf[:len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep:]
                    break
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(self.OPEN):]
                self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._in_think else self._buf
        self._buf = ""
        return rest


_HARD_TERMINATORS = "।॥\n"
_SOFT_TERMINATORS = ".?!"
_TRAILING_CLOSERS = "\"'”’)]»"


def split_sentences(text: str, strip_citation_marks: bool = True) -> List[str]:
    """
    Splits an answer into sentences for TTS chunking.
    Boundaries: '।', '॥', newlines, and '.', '?', '!' when followed by whitespace / end of text
    (so decimals like 3.5 and abbreviations like એક્સ.આર.સી stay intact; list numbers like
    "1." do not form their own chunk). Citation markers and markdown symbols are removed.
    """
    if not text:
        return []
    if strip_citation_marks:
        text = strip_citations(text)
    text = strip_think(text).replace("\r", "")

    pieces: List[str] = []
    buf: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        boundary = False
        if ch in _HARD_TERMINATORS:
            boundary = True
        elif ch in _SOFT_TERMINATORS:
            j = i + 1
            while j < n and (text[j] in _SOFT_TERMINATORS or text[j] in _TRAILING_CLOSERS):
                buf.append(text[j])
                j += 1
            i = j - 1
            at_gap = j >= n or text[j].isspace()
            current = "".join(buf).strip()
            is_list_number = ch == "." and current[:-1].strip().isdigit()
            boundary = at_gap and not is_list_number
        if boundary:
            pieces.append("".join(buf))
            buf = []
        i += 1
    if buf:
        pieces.append("".join(buf))

    sentences = []
    for p in pieces:
        p = re.sub(r"[*_`#]{1,3}", "", p)
        p = re.sub(r"^\s*(?:[-•]\s+)", "", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p and any(c.isalnum() for c in p):
            sentences.append(p)
    return sentences


def _normalize_for_match(text: str) -> str:
    text = strip_citations(strip_think(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"'“”‘’«»*_` ").rstrip(".।!॥ ").strip().lower()


_NO_ANSWER_NORMALIZED = [_normalize_for_match(m) for m in NO_ANSWER_MESSAGES.values()]


def is_no_answer(answer: str) -> bool:
    """True when the answer is (essentially only) one of the refusal sentences, or empty."""
    a = _normalize_for_match(answer)
    if not a:
        return True
    return any(r in a and len(a) <= len(r) + 25 for r in _NO_ANSWER_NORMALIZED)


# ------------------------------------------------------------
# Reasoning-model parameters (with graceful fallback when Groq rejects one)
# ------------------------------------------------------------

_REJECTED_PARAMS = set()
_REJECTED_LOCK = threading.Lock()


def _is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in (model or "").lower()


def _is_qwen3(model: str) -> bool:
    m = (model or "").lower()
    return "qwen3" in m or "qwq" in m


def resolve_reasoning_effort(model: str, effort: Optional[str] = None) -> Optional[str]:
    """Maps the configured effort to a value the model accepts (None = do not send)."""
    e = (effort or LLM_REASONING_EFFORT or "auto").strip().lower()
    if e in ("off", "unset", ""):
        return None
    if e == "auto":
        if _is_gpt_oss(model):
            return "low"
        if _is_qwen3(model):
            return "none"
        return None
    if _is_gpt_oss(model) and e in ("none", "default"):
        return "low"  # gpt-oss cannot switch reasoning off
    return e


def _param_rejected(model: str, param: str) -> bool:
    with _REJECTED_LOCK:
        return (model, param) in _REJECTED_PARAMS


def reasoning_extra_body(model: str, effort: Optional[str] = None) -> Dict[str, Any]:
    """Groq reasoning parameters to send for `model` (skips ones Groq rejected earlier)."""
    extra: Dict[str, Any] = {}
    eff = resolve_reasoning_effort(model, effort)
    if eff and not _param_rejected(model, "reasoning_effort"):
        extra["reasoning_effort"] = eff
    fmt = (LLM_REASONING_FORMAT or "").strip().lower()
    if fmt and not _is_gpt_oss(model) and not _param_rejected(model, "reasoning_format"):
        extra["reasoning_format"] = fmt
    return extra


def _error_text(err: Exception) -> str:
    parts = [str(err)]
    body = getattr(err, "body", None)
    if body:
        parts.append(str(body))
    return " ".join(parts).lower()


def create_chat_completion(client, *, model: str, messages: List[Dict[str, str]],
                           reasoning_effort: Optional[str] = None, **kwargs):
    """
    `client.chat.completions.create` plus reasoning parameters. If Groq rejects a reasoning
    parameter (HTTP 400 naming it), the call is retried once without it and the rejection is
    remembered for that model, so later calls do not pay for the failed attempt again.
    """
    body: Dict[str, Any] = dict(kwargs.pop("extra_body", None) or {})
    extra = reasoning_extra_body(model, reasoning_effort)
    body.update(extra)
    try:
        return client.chat.completions.create(model=model, messages=messages, extra_body=body or None, **kwargs)
    except Exception as e:
        if getattr(e, "status_code", None) != 400 or not extra:
            raise
        msg = _error_text(e)
        bad = [p for p in extra if p in msg or p.replace("_", " ") in msg]
        if not bad and "reasoning" in msg:
            bad = list(extra)
        if not bad:
            raise
        with _REJECTED_LOCK:
            for p in bad:
                _REJECTED_PARAMS.add((model, p))
        for p in bad:
            body.pop(p, None)
        print(f"⚠️ [generation] {model} rejected {bad}; retrying without them (remembered).")
        return client.chat.completions.create(model=model, messages=messages, extra_body=body or None, **kwargs)


# ------------------------------------------------------------
# Prompt construction & conversation memory
# ------------------------------------------------------------

def _get_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def normalize_history(
    history: Optional[Iterable[Any]],
    max_turns: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Cleans conversation history (dicts or objects with role/content), oldest first.
    Keeps only user/assistant messages with text, the last `max_turns` messages
    (RAG_HISTORY_TURNS), each truncated to `max_chars` (RAG_HISTORY_MAX_CHARS).
    """
    max_turns = HISTORY_TURNS if max_turns is None else max_turns
    max_chars = HISTORY_MAX_CHARS if max_chars is None else max_chars
    if not history or max_turns <= 0:
        return []
    cleaned: List[Dict[str, str]] = []
    for item in history:
        role = str(_get_field(item, "role") or "").strip().lower()
        content = _get_field(item, "content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "…"
        cleaned.append({"role": role, "content": content})
    return cleaned[-max_turns:]


def build_context(contexts: List[str], lang: str = "gu") -> str:
    code = normalize_answer_lang(lang)
    tag = ANSWER_LANGUAGE_META[code]["context_label"]
    return "\n\n".join(f"[{tag} {rank}] {text}" for rank, text in enumerate(contexts, start=1))


def build_user_prompt(query: str, context: str, lang: str = "gu") -> str:
    meta = ANSWER_LANGUAGE_META[normalize_answer_lang(lang)]
    return (
        f"{meta['context_header']}:\n{context}\n\n"
        f"{meta['question_header']}:\n{query}\n\n"
        f"{meta['answer_header']}:"
    )


def build_messages(
    query: str,
    contexts: List[str],
    lang: str = "gu",
    history: Optional[Iterable[Any]] = None,
    answer_lang: Optional[str] = None,
) -> List[Dict[str, str]]:
    """System prompt (answer language) + last RAG_HISTORY_TURNS messages + final user turn with context."""
    code = normalize_answer_lang(answer_lang or lang)
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[code]}]
    for m in normalize_history(history):
        content = strip_citations(m["content"]) if m["role"] == "assistant" else m["content"]
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": build_user_prompt(query, build_context(contexts, code), code)})
    return messages


_CONDENSE_PREFIX_RE = re.compile(
    r"^\s*(?:standalone(?:\s+search)?\s+question|rewritten\s+question|question|પ્રશ્ન|प्रश्न)\s*[:：-]\s*",
    re.IGNORECASE,
)


def _clean_condensed(text: str) -> str:
    text = strip_think(text or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = _CONDENSE_PREFIX_RE.sub("", lines[0])
    return out.strip().strip("\"'“”‘’`*").strip()


@traceable(run_type="chain", name="condense_query")
def condense_query(
    query: str,
    history: Optional[Iterable[Any]],
    lang: str = "gu",
    model: Optional[str] = None,
    client: Any = None,
) -> str:
    """
    Rewrites a follow-up question into a standalone question in the same language, using the
    conversation history (one short, low-cost LLM call: small max_tokens, no reasoning, no
    retries). Returns the original query on any error or implausible output.
    """
    hist = normalize_history(history)
    if not hist or not (query or "").strip():
        return query
    code = normalize_answer_lang(lang)
    lang_name = ANSWER_LANGUAGE_META[code]["name"]
    system = (
        "You rewrite the user's latest follow-up question into a standalone search question. "
        f"Write it in {lang_name}, the same language as the follow-up question. Resolve pronouns and "
        "references (he, she, it, that, there, ...) using the conversation. Keep the meaning; do not answer "
        "it and do not add new facts. If the follow-up is already standalone, return it unchanged. "
        "Output only the question, nothing else."
    )
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {strip_citations(m['content'])[:600]}" for m in hist
    )
    user = f"Conversation:\n{convo}\n\nFollow-up question: {query}\n\nStandalone question:"
    try:
        resp = create_chat_completion(
            client or get_fast_llm_client(),
            model=model or CONDENSE_MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=CONDENSE_MAX_TOKENS,
            timeout=CONDENSE_TIMEOUT,
        )
        text = _clean_condensed(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"⚠️ [generation] Query condensing failed, using the original query: {e}")
        return query

    if not text or len(text) > max(400, 4 * len(query)):
        return query
    if code in ("gu", "hi") and detect_script_language(text) != code:
        return query  # model switched language/script: keep the user's own words
    return text


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

def _final_item(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return items[-1] if items else {}


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"\S+\s*", text)


def _delta_reasoning(delta: Any) -> str:
    for name in ("reasoning", "reasoning_content"):
        value = getattr(delta, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


@traceable(run_type="chain", name="llm_generate", reduce_fn=_final_item)
def stream_answer(
    query: str,
    contexts: List[str],
    lang: str = "gu",
    model: Optional[str] = None,
    history: Optional[Iterable[Any]] = None,
    answer_lang: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams the grounded answer. Yields {"type": "token", "content": str} per delta and a final
    {"type": "done", "answer", "ttft_ms", "llm_ms", "model", "no_answer", "answer_language"} item.

    - No contexts -> the refusal sentence of the answer language is streamed without an LLM call.
    - Hidden reasoning (<think>...</think>) is stripped from the streamed content; reasoning
      models get `reasoning_effort` (none for Qwen3, low for gpt-oss) so they answer directly.
    - `no_answer` is True when the model's answer is the refusal sentence (or empty).
    The underlying Groq call is traced as an `llm` run by wrap_openai.
    """
    model = model or GROQ_MODEL_NAME
    code = normalize_answer_lang(answer_lang or lang)
    t0 = time.perf_counter()
    contexts = [c for c in (contexts or []) if isinstance(c, str) and c.strip()]

    if not contexts:
        refusal = NO_ANSWER_MESSAGES[code]
        ttft_ms = 0.0
        for idx, tok in enumerate(_text_tokens(refusal)):
            if idx == 0:
                ttft_ms = (time.perf_counter() - t0) * 1000
            yield {"type": "token", "content": tok}
        yield {
            "type": "done",
            "answer": refusal,
            "ttft_ms": ttft_ms,
            "llm_ms": 0.0,
            "model": None,
            "no_answer": True,
            "answer_language": code,
        }
        return

    stream = create_chat_completion(
        get_llm_client(),
        model=model,
        messages=build_messages(query, contexts, lang, history=history, answer_lang=code),
        temperature=LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        stream=True,
    )

    ttft_ms = 0.0
    parts: List[str] = []
    reasoning_chars = 0
    finish_reason = None
    think_filter = ThinkStreamFilter()

    def _emit(visible: str):
        nonlocal ttft_ms
        if not parts:
            visible = visible.lstrip()
            if not visible:
                return None
            ttft_ms = (time.perf_counter() - t0) * 1000
        if not visible:
            return None
        parts.append(visible)
        return {"type": "token", "content": visible}

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        delta = choice.delta
        if delta is None:
            continue
        reasoning_chars += len(_delta_reasoning(delta))
        raw = delta.content or ""
        if not raw:
            continue
        item = _emit(think_filter.feed(raw))
        if item:
            yield item
    item = _emit(think_filter.flush())
    if item:
        yield item

    answer = strip_think("".join(parts)).strip()
    no_answer = is_no_answer(answer)
    if not answer:
        # Nothing visible (e.g. the token budget went to reasoning): answer with the refusal sentence.
        answer = NO_ANSWER_MESSAGES[code]
        for tok in _text_tokens(answer):
            if not parts:
                ttft_ms = (time.perf_counter() - t0) * 1000
            parts.append(tok)
            yield {"type": "token", "content": tok}

    yield {
        "type": "done",
        "answer": answer,
        "ttft_ms": ttft_ms,
        "llm_ms": (time.perf_counter() - t0) * 1000,
        "model": model,
        "no_answer": no_answer,
        "answer_language": code,
        "finish_reason": finish_reason,
        "reasoning_chars": reasoning_chars,
    }


def generate_answer(
    query: str,
    contexts: List[str],
    lang: str = "gu",
    model: Optional[str] = None,
    history: Optional[Iterable[Any]] = None,
    answer_lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-streaming convenience wrapper around `stream_answer`."""
    result: Dict[str, Any] = {}
    for item in stream_answer(query, contexts, lang=lang, model=model, history=history, answer_lang=answer_lang):
        if item["type"] == "done":
            result = item
    result.pop("type", None)
    return result


def warmup_llm() -> None:
    """Opens the pooled TLS connections to Groq so the first user request does not pay for them."""
    try:
        get_llm_client().models.list()
        get_fast_llm_client().models.list()
    except Exception as e:
        print(f"⚠️ [generation] Groq warm-up failed: {e}")
