"""
Multilingual Prompting & Generation Layer
=========================================
Centralized system prompts, context construction, and Groq LLM streaming.
"""

import os
import time
from typing import Any, Dict, Generator, List, Optional

from core.config import (
    GROQ_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPTS,
    LANGUAGE_METADATA,
)
from core.tracing import get_llm_client, traceable


def normalize_lang_code(lang: Optional[str]) -> str:
    """Normalizes any language string/alias to canonical code ('gu' or 'hi')."""
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


def detect_script_language(text: str) -> Optional[str]:
    """
    Detects language based on Unicode script distribution:
    - Gujarati: U+0A80 to U+0AFF
    - Hindi (Devanagari): U+0900 to U+097F
    Returns 'gu', 'hi', or None if no Indic characters detected.
    """
    if not text:
        return None
    gu_count = sum(1 for ch in text if "\u0a80" <= ch <= "\u0aff")
    hi_count = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    if gu_count > hi_count and gu_count > 0:
        return "gu"
    if hi_count > gu_count and hi_count > 0:
        return "hi"
    return None


def build_context(contexts: List[str], lang: str = "gu") -> str:
    code = normalize_lang_code(lang)
    tag = LANGUAGE_METADATA[code]["context_header"]
    return "\n\n".join(f"[{tag} {rank}] {text}" for rank, text in enumerate(contexts, start=1))


def build_user_prompt(query: str, context: str, lang: str = "gu") -> str:
    code = normalize_lang_code(lang)
    meta = LANGUAGE_METADATA[code]
    return (
        f"{meta['context_header']}:\n{context}\n\n"
        f"{meta['question_header']}:\n{query}\n\n"
        f"{meta['answer_header']}:"
    )


def build_messages(query: str, contexts: List[str], lang: str = "gu") -> List[Dict[str, str]]:
    code = normalize_lang_code(lang)
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[code]},
        {"role": "user", "content": build_user_prompt(query, build_context(contexts, code), code)},
    ]


def _final_item(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return items[-1] if items else {}


@traceable(run_type="chain", name="llm_generate", reduce_fn=_final_item)
def stream_answer(
    query: str,
    contexts: List[str],
    lang: str = "gu",
    model: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams the grounded answer. Yields {"type": "token", "content": str} per delta and
    a final {"type": "done", "answer", "ttft_ms", "llm_ms", "model"} item.
    The underlying Groq call is traced as an `llm` run by wrap_openai.
    """
    model = model or GROQ_MODEL_NAME
    t0 = time.perf_counter()
    stream = get_llm_client().chat.completions.create(
        model=model,
        messages=build_messages(query, contexts, lang),
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        stream=True,
    )

    ttft_ms = 0.0
    parts: List[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            if not parts:
                ttft_ms = (time.perf_counter() - t0) * 1000
            parts.append(delta)
            yield {"type": "token", "content": delta}

    yield {
        "type": "done",
        "answer": "".join(parts).strip(),
        "ttft_ms": ttft_ms,
        "llm_ms": (time.perf_counter() - t0) * 1000,
        "model": model,
    }


def generate_answer(query: str, contexts: List[str], lang: str = "gu", model: Optional[str] = None) -> Dict[str, Any]:
    """Non-streaming convenience wrapper around `stream_answer`."""
    result: Dict[str, Any] = {}
    for item in stream_answer(query, contexts, lang=lang, model=model):
        if item["type"] == "done":
            result = item
    result.pop("type", None)
    return result


def warmup_llm() -> None:
    """Opens the pooled TLS connection to Groq so the first user request does not pay for it."""
    try:
        get_llm_client().models.list()
    except Exception as e:
        print(f"⚠️ [generation] Groq warm-up failed: {e}")
