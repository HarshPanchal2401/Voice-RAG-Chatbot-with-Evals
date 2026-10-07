"""
LLM-as-a-judge backend (shared by the offline harness and the online evaluator)
==============================================================================
* ONE fixed judge model per run - no silent fallback to other models.
* Retries 429 / 5xx / timeouts / connection errors with exponential backoff + jitter
  (honours ``Retry-After``); raises :class:`JudgeError` on final failure - it never
  returns fake JSON.
* Real async ``a_generate`` (``groq.AsyncGroq``) so DeepEval's async metrics run
  concurrently.
* Token-bucket rate limiting shared by sync and async callers.
* Optional on-disk cache keyed by sha256(model, messages, params).
* Records which model actually answered and per-run call statistics.
* For reasoning models (``openai/gpt-oss-*``) requests low reasoning effort and checks
  ``finish_reason`` so reasoning tokens cannot silently truncate the JSON verdict.

Model roles (generator / reranker / judge) are resolved from ``core.config`` with
environment fallbacks and must be distinct (see :func:`check_model_roles`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# DeepEval sends anonymous telemetry by default; evaluation must not make hidden network calls.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402

DEFAULT_GENERATOR_MODEL = "qwen/qwen3.8-27b"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_RERANKER_BACKEND = "llm"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluation judge for a Gujarati/Hindi/English "
    "question-answering system. Texts may be in Gujarati, Hindi or English; judge meaning, "
    "not script or wording. Output ONLY valid JSON that follows the requested schema."
)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class JudgeError(RuntimeError):
    """The judge could not produce a usable answer (after retries)."""


class ModelRoleError(ValueError):
    """Judge model collides with the generator or the reranker model."""


# ------------------------------------------------------------------
# Model roles
# ------------------------------------------------------------------

@dataclass
class ModelRoles:
    generator: str
    judge: str
    reranker_backend: str
    reranker_model: Optional[str]
    source: str = "env"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_model_roles(
    judge_override: Optional[str] = None,
    use_core_config: bool = True,
) -> ModelRoles:
    """
    Generator / reranker / judge names from ``core.config`` (GROQ_MODEL_NAME,
    RERANKER_MODEL_NAME, DEFAULT_JUDGE_MODEL, RERANKER_BACKEND), falling back to the
    env vars RAG_LLM_MODEL / RAG_RERANKER_MODEL / RAG_JUDGE_MODEL / RAG_RERANKER_BACKEND.
    """
    gen = os.environ.get("RAG_LLM_MODEL", DEFAULT_GENERATOR_MODEL)
    judge = os.environ.get("RAG_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    rr_model = os.environ.get("RAG_RERANKER_MODEL")
    rr_backend = os.environ.get("RAG_RERANKER_BACKEND", DEFAULT_RERANKER_BACKEND)
    source = "env"
    if use_core_config:
        try:
            import core.config as cc  # noqa: WPS433 (import inside function on purpose)

            gen = getattr(cc, "GROQ_MODEL_NAME", gen) or gen
            judge = getattr(cc, "DEFAULT_JUDGE_MODEL", judge) or judge
            rr_model = getattr(cc, "RERANKER_MODEL_NAME", rr_model) or rr_model
            rr_backend = getattr(cc, "RERANKER_BACKEND", rr_backend) or rr_backend
            source = "core.config"
        except Exception:  # pragma: no cover - config import problems fall back to env
            source = "env (core.config import failed)"
    if judge_override:
        judge = judge_override
    return ModelRoles(
        generator=str(gen),
        judge=str(judge),
        reranker_backend=str(rr_backend).lower(),
        reranker_model=str(rr_model) if rr_model else None,
        source=source,
    )


def model_role_problems(roles: ModelRoles) -> List[str]:
    problems = []
    if roles.judge.strip().lower() == roles.generator.strip().lower():
        problems.append(
            f"Judge model '{roles.judge}' is the same as the generator model '{roles.generator}' "
            "(self-judging inflates scores). Set RAG_JUDGE_MODEL to a different model."
        )
    if (
        roles.reranker_backend == "llm"
        and roles.reranker_model
        and roles.judge.strip().lower() == roles.reranker_model.strip().lower()
    ):
        problems.append(
            f"Judge model '{roles.judge}' is the same as the LLM reranker model '{roles.reranker_model}'. "
            "Set RAG_JUDGE_MODEL or RAG_RERANKER_MODEL so they differ."
        )
    return problems


def check_model_roles(roles: ModelRoles) -> ModelRoles:
    """Raise :class:`ModelRoleError` when the judge collides with generator/reranker."""
    problems = model_role_problems(roles)
    if problems:
        raise ModelRoleError(" ".join(problems))
    return roles


# ------------------------------------------------------------------
# JSON extraction
# ------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```")


def _first_balanced_object(text: str) -> Optional[str]:
    """First balanced {...} block, ignoring braces inside JSON strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json_object(text: Optional[str]) -> Dict[str, Any]:
    """
    Parse the JSON object out of a judge reply that may contain <think> blocks,
    markdown fences or surrounding prose. Raises ValueError when no valid object exists.
    """
    if text is None:
        raise ValueError("empty judge output")
    cleaned = _THINK_RE.sub("", str(text)).strip()
    candidates: List[str] = []
    for m in _FENCE_RE.finditer(cleaned):
        candidates.append(m.group(1).strip())
    candidates.append(cleaned)
    for cand in candidates:
        block = _first_balanced_object(cand)
        if not block:
            continue
        for attempt in (block, re.sub(r",\s*([\]}])", r"\1", block)):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"no valid JSON object in judge output: {cleaned[:200]!r}")


# ------------------------------------------------------------------
# Rate limiting + cache
# ------------------------------------------------------------------

class TokenBucket:
    """Token bucket usable from threads and coroutines. rate = requests per minute."""

    def __init__(self, rate_per_min: float = 30.0, burst: Optional[int] = None):
        self.rate = max(0.001, float(rate_per_min)) / 60.0
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_min // 6) or 1))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def _reserve(self) -> float:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            self.tokens -= 1.0
            if self.tokens >= 0:
                return 0.0
            return -self.tokens / self.rate

    def acquire(self) -> None:
        wait = self._reserve()
        if wait > 0:
            time.sleep(wait)

    async def a_acquire(self) -> None:
        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)


class JudgeCache:
    """On-disk cache: <dir>/<sha[:2]>/<sha>.json. Writes are atomic."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, p)


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".judge_cache"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------
# The judge model
# ------------------------------------------------------------------

class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Groq with retries, async, rate limiting and caching."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        max_retries: Optional[int] = None,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        rate_limiter: Optional[TokenBucket] = None,
        cache_dir: Optional[Path | str] = None,
        use_cache: bool = False,
        client: Any = None,
        async_client: Any = None,
    ):
        self.model_name = model_name or resolve_model_roles(use_core_config=False).judge
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.max_retries = int(max_retries if max_retries is not None else _env_float("RAG_JUDGE_MAX_RETRIES", 5))
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = float(timeout if timeout is not None else _env_float("RAG_JUDGE_TIMEOUT", 60.0))
        self.max_tokens = int(max_tokens if max_tokens is not None else _env_float("RAG_JUDGE_MAX_TOKENS", 4096))
        self.reasoning_effort = reasoning_effort or os.environ.get("RAG_JUDGE_REASONING_EFFORT", "low")
        self.rate_limiter = rate_limiter or TokenBucket(_env_float("RAG_JUDGE_RPM", 30.0))
        self.cache = JudgeCache(Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR) if use_cache else None
        self._client = client
        self._async_client = None
        self._async_loop = None
        self._injected_async_client = async_client
        self._rng = random.Random()
        self._lock = threading.Lock()
        self.answered_by: Counter = Counter()
        self.stats: Counter = Counter()
        super().__init__(self.model_name)

    # -- DeepEval interface -------------------------------------------------
    def load_model(self):
        if self._client is None and self.api_key:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        return self._client  # None when no API key: every call then raises JudgeError

    def get_model_name(self) -> str:
        return self.model_name

    @property
    def async_client(self):
        # httpx async clients are bound to the event loop they were first used in;
        # keep one client per loop so repeated asyncio.run() calls keep working.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._async_client is not None and self._async_loop is loop:
            return self._async_client
        if self._injected_async_client is not None:
            return self._injected_async_client
        if not self.api_key:
            raise JudgeError("GROQ_API_KEY is not set; the judge cannot be called.")
        from groq import AsyncGroq

        self._async_client = AsyncGroq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        self._async_loop = loop
        return self._async_client

    # -- helpers --------------------------------------------------------------
    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] += n

    def _is_reasoning_model(self) -> bool:
        return self.model_name.lower().startswith("openai/gpt-oss")

    def _build_request(self, prompt: str, schema: Any, max_tokens: int) -> Dict[str, Any]:
        wants_json = schema is not None or "json" in prompt.lower()
        messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        req: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if wants_json:
            req["response_format"] = {"type": "json_object"}
        if self._is_reasoning_model() and self.reasoning_effort:
            req["reasoning_effort"] = self.reasoning_effort
        return req

    @staticmethod
    def cache_key(req: Dict[str, Any]) -> str:
        payload = {k: req[k] for k in sorted(req) if k != "max_tokens"}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _delay(self, attempt: int, exc: Optional[BaseException]) -> float:
        retry_after = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                retry_after = float(resp.headers.get("retry-after"))
            except Exception:
                retry_after = None
        backoff = min(self.max_delay, self.base_delay * (2 ** attempt))
        jittered = backoff * (0.5 + self._rng.random())
        if retry_after is not None:
            return min(self.max_delay * 2, max(retry_after, jittered))
        return jittered

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        try:
            import groq
        except Exception:  # pragma: no cover
            groq = None
        if groq is not None:
            if isinstance(exc, (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError)):
                return True
            if isinstance(exc, groq.APIStatusError):
                return getattr(exc, "status_code", None) in RETRYABLE_STATUS
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status in RETRYABLE_STATUS
        return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))

    def _postprocess(self, resp: Any, schema: Any, wants_json: bool) -> Any:
        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        content = (getattr(choice.message, "content", None) or "").strip()
        if finish == "length":
            self._bump("truncations")
            raise _Truncated(f"judge output truncated (finish_reason=length, {len(content)} chars)")
        answered = getattr(resp, "model", None) or self.model_name
        with self._lock:
            self.answered_by[answered] += 1
        if not wants_json:
            return _THINK_RE.sub("", content).strip(), answered
        try:
            obj = extract_json_object(content)
        except ValueError as e:
            self._bump("invalid_json")
            raise _InvalidJSON(str(e))
        return obj, answered

    def _finalize(self, obj: Any, schema: Any) -> Any:
        if isinstance(obj, str):
            return obj
        if schema is not None:
            try:
                return schema.model_validate(obj)
            except Exception:
                pass  # let DeepEval's own (lenient) JSON handling try the raw JSON
        return json.dumps(obj, ensure_ascii=False)

    # -- sync ------------------------------------------------------------------
    def generate(self, prompt: str, schema: Any = None, *args, **kwargs) -> Any:
        wants_json = schema is not None or "json" in str(prompt).lower()
        max_tokens = self.max_tokens
        req = self._build_request(prompt, schema, max_tokens)
        key = self.cache_key(req)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self._bump("cache_hits")
                with self._lock:
                    self.answered_by[hit.get("answered_by", self.model_name)] += 1
                return self._finalize(hit["output"], schema)
        if self.model is None:
            self._bump("failures")
            raise JudgeError("GROQ_API_KEY is not set; the judge cannot be called.")
        format_retries = 1
        last_exc: Optional[BaseException] = None
        attempt = 0
        while attempt <= self.max_retries:
            self.rate_limiter.acquire()
            self._bump("calls")
            try:
                resp = self.model.chat.completions.create(**req)
                out, answered = self._postprocess(resp, schema, wants_json)
                if self.cache is not None:
                    self.cache.set(key, {"model": self.model_name, "answered_by": answered, "output": out})
                return self._finalize(out, schema)
            except (_Truncated, _InvalidJSON) as e:
                last_exc = e
                if format_retries <= 0:
                    break
                format_retries -= 1
                if isinstance(e, _Truncated):
                    req["max_tokens"] = min(16384, int(req["max_tokens"]) * 2)
                continue
            except Exception as e:  # network / API errors
                last_exc = e
                if not self._is_retryable(e) or attempt >= self.max_retries:
                    break
                self._bump("retries")
                time.sleep(self._delay(attempt, e))
                attempt += 1
        self._bump("failures")
        raise JudgeError(f"judge '{self.model_name}' failed: {type(last_exc).__name__}: {last_exc}") from last_exc

    # -- async -----------------------------------------------------------------
    async def a_generate(self, prompt: str, schema: Any = None, *args, **kwargs) -> Any:
        wants_json = schema is not None or "json" in str(prompt).lower()
        req = self._build_request(prompt, schema, self.max_tokens)
        key = self.cache_key(req)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self._bump("cache_hits")
                with self._lock:
                    self.answered_by[hit.get("answered_by", self.model_name)] += 1
                return self._finalize(hit["output"], schema)
        format_retries = 1
        last_exc: Optional[BaseException] = None
        attempt = 0
        while attempt <= self.max_retries:
            await self.rate_limiter.a_acquire()
            self._bump("calls")
            try:
                resp = await self.async_client.chat.completions.create(**req)
                out, answered = self._postprocess(resp, schema, wants_json)
                if self.cache is not None:
                    self.cache.set(key, {"model": self.model_name, "answered_by": answered, "output": out})
                return self._finalize(out, schema)
            except (_Truncated, _InvalidJSON) as e:
                last_exc = e
                if format_retries <= 0:
                    break
                format_retries -= 1
                if isinstance(e, _Truncated):
                    req["max_tokens"] = min(16384, int(req["max_tokens"]) * 2)
                continue
            except Exception as e:
                last_exc = e
                if not self._is_retryable(e) or attempt >= self.max_retries:
                    break
                self._bump("retries")
                await asyncio.sleep(self._delay(attempt, e))
                attempt += 1
        self._bump("failures")
        raise JudgeError(f"judge '{self.model_name}' failed: {type(last_exc).__name__}: {last_exc}") from last_exc

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "answered_by": dict(self.answered_by),
            "stats": dict(self.stats),
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort if self._is_reasoning_model() else None,
            "cache": str(self.cache.dir) if self.cache else None,
        }


class _Truncated(Exception):
    pass


class _InvalidJSON(Exception):
    pass


# Backwards-compatible name used by older imports.
GroqDeepEvalModel = GroqJudge
