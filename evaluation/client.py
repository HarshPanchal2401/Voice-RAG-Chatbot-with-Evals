"""
RAG Service Client (explicit mode, no silent fallback)
======================================================
mode="server"    : talks to the running API (``--base-url``). Sends ``X-API-Key`` from the
                   env var ``RAG_API_KEY`` when set. 429 / 5xx are retried honouring
                   ``Retry-After``; any other failure raises ``RAGClientError``.
mode="inprocess" : calls a ``LanguageRouter`` passed in by the caller (same
                   ``pipeline.ask`` / ``hybrid_retrieve`` the API uses).

A run never mixes the two modes: if the server is unreachable the run fails with a
clear message instead of switching to in-process retrieval.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

RETRY_STATUS = {429, 500, 502, 503, 504}


class RAGClientError(RuntimeError):
    pass


class RAGClient:
    def __init__(
        self,
        mode: str = "server",
        base_url: str = "http://localhost:8000",
        router: Optional[Any] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        fallback_router: Optional[Any] = None,  # deprecated alias of `router`
    ):
        if mode not in ("server", "inprocess"):
            raise ValueError(f"mode must be 'server' or 'inprocess', got {mode!r}")
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.router = router or fallback_router
        self.api_key = api_key if api_key is not None else os.environ.get("RAG_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self._health: Optional[Dict[str, Any]] = None
        if self.mode == "inprocess" and self.router is None:
            raise RAGClientError("mode='inprocess' needs a LanguageRouter (pass router=...).")

    # ------------------------------------------------------------------ http
    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "RAG-Eval"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                last = f"HTTP {e.code} {body}"
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    try:
                        wait = float(retry_after) if retry_after else min(30.0, 2.0 ** attempt)
                    except ValueError:
                        wait = min(30.0, 2.0 ** attempt)
                    time.sleep(wait)
                    continue
                if e.code == 401:
                    raise RAGClientError(f"{method} {url} -> 401 Unauthorized. Set RAG_API_KEY for the evaluation run.")
                raise RAGClientError(f"{method} {url} failed: {last}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = str(e)
                if attempt < self.max_retries:
                    time.sleep(min(30.0, 2.0 ** attempt))
                    continue
                raise RAGClientError(f"{method} {url} failed: {last}")
        raise RAGClientError(f"{method} {url} failed after retries: {last}")

    # ------------------------------------------------------------------ api
    def check(self) -> Dict[str, Any]:
        """Verifies the backend is usable; raises RAGClientError otherwise."""
        if self.mode == "inprocess":
            self._health = {"mode": "inprocess"}
            return self._health
        try:
            self._health = self._request("GET", "/health", timeout=10)
        except RAGClientError as e:
            raise RAGClientError(
                f"RAG server not reachable at {self.base_url} ({e}). Start the API or run with --mode inprocess."
            )
        return self._health

    @property
    def health(self) -> Dict[str, Any]:
        if self._health is None:
            self.check()
        return self._health or {}

    def is_server_online(self) -> bool:  # backwards compatibility
        if self.mode != "server":
            return False
        try:
            self.check()
            return True
        except RAGClientError:
            return False

    def retrieve(self, query: str, language: str = "gu", top_k: int = 5, use_reranker: bool = False,
                 sort_by: str = "rrf") -> Dict[str, Any]:
        t0 = time.perf_counter()
        if self.mode == "server":
            data = self._request("POST", "/api/v1/retrieve", {
                "query": query, "language": language, "top_k": top_k,
                "use_reranker": use_reranker, "sort_by": sort_by, "auto_detect_language": False,
            })
            client_ms = (time.perf_counter() - t0) * 1000
            timings = dict(data.get("retrieval_timings") or {})
            return {"documents": data.get("sources") or [], "timings": timings,
                    "language": data.get("language", language), "client_ms": client_ms}
        pipe = self.router.get_pipeline(language)
        res = pipe.hybrid_retrieve(query, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by)
        client_ms = (time.perf_counter() - t0) * 1000
        return {"documents": res.get("documents") or [], "timings": dict(res.get("timings") or {}),
                "language": res.get("language", language), "client_ms": client_ms}

    def generate(self, query: str, language: str = "gu", top_k: int = 5, use_reranker: bool = False,
                 sort_by: str = "rrf") -> Dict[str, Any]:
        """Full production path (retrieve + generate). Raises RAGClientError on failure."""
        t0 = time.perf_counter()
        if self.mode == "server":
            data = self._request("POST", "/api/v1/query/text", {
                "query": query, "language": language, "top_k": top_k, "use_reranker": use_reranker,
                "sort_by": sort_by, "auto_detect_language": False, "evaluate": False, "voice_reply": False,
            })
            client_ms = (time.perf_counter() - t0) * 1000
            lat = data.get("latency") or {}
            return {
                "answer": data.get("answer", ""),
                "documents": data.get("sources") or [],
                "retrieval_timings": lat.get("retrieval") or {},
                "llm_ms": lat.get("llm_ms"),
                "ttft_ms": lat.get("ttft_ms"),
                "total_ms": lat.get("total_ms"),
                "client_ms": client_ms,
                "trace_id": data.get("trace_id"),
                "no_answer": data.get("no_answer"),
                "answer_language": data.get("answer_language"),
                "model": data.get("model") or self.health.get("llm_model"),
                "language": data.get("language", language),
            }
        pipe = self.router.get_pipeline(language)
        res = pipe.ask(query, final_k=top_k, use_reranker=use_reranker, sort_by=sort_by)
        client_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": res.get("answer", ""),
            "documents": res.get("documents") or [],
            "retrieval_timings": res.get("retrieval_timings") or {},
            "llm_ms": res.get("llm_ms"),
            "ttft_ms": res.get("ttft_ms"),
            "total_ms": res.get("total_ms"),
            "client_ms": client_ms,
            "trace_id": res.get("trace_id"),
            "no_answer": res.get("no_answer"),
            "answer_language": res.get("answer_language"),
            "model": res.get("model"),
            "language": res.get("language", language),
        }

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"mode": self.mode}
        if self.mode == "server":
            info["base_url"] = self.base_url
            info["auth"] = bool(self.api_key)
            h = self._health or {}
            for k in ("reranker_backend", "llm_model", "version", "auth_required"):
                if k in h:
                    info[f"server_{k}"] = h[k]
        return info


def make_inprocess_router():
    """Loads the full LanguageRouter (heavy: BGE-M3 + FAISS indexes)."""
    from pipeline.router import LanguageRouter

    return LanguageRouter(verbose=False)
