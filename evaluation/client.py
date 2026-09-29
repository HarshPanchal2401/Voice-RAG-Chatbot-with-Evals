"""
RAG Service Client
==================
Communicates with the live RAG server (http://localhost:8000) or seamlessly
falls back to in-process LanguageRouter if the server is offline.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional


class RAGClient:
    """Connects to FastAPI Voice RAG endpoints or in-memory router."""

    def __init__(self, base_url: str = "http://localhost:8000", fallback_router: Optional[Any] = None):
        self.base_url = base_url.rstrip("/")
        self.fallback_router = fallback_router
        self._server_online: Optional[bool] = None

    def is_server_online(self) -> bool:
        if self._server_online is not None:
            return self._server_online
        try:
            req = urllib.request.Request(f"{self.base_url}/health", headers={"User-Agent": "RAG-Eval"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._server_online = (resp.status == 200)
        except Exception:
            self._server_online = False
        return self._server_online

    def retrieve(self, query: str, language: str = "gu", top_k: int = 5) -> Dict[str, Any]:
        """Retrieves passages via HTTP or in-process router."""
        if self.is_server_online():
            url = f"{self.base_url}/api/v1/retrieve"
            payload = json.dumps({
                "query": query,
                "language": language,
                "top_k": top_k
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    timings = data.get("retrieval_timings", {})
                    timings["retrieval_total_ms"] = elapsed_ms
                    return {
                        "documents": data.get("sources", []),
                        "timings": timings,
                        "language": language
                    }
            except Exception as e:
                print(f"⚠️ API retrieve failed ({e}), falling back to in-process router...")

        # In-process fallback
        if self.fallback_router:
            pipeline = self.fallback_router.get_pipeline(language)
            return pipeline.hybrid_retrieve(query, final_k=top_k)

        raise RuntimeError("Neither live RAG server nor in-process router is available.")

    def generate(self, query: str, language: str = "gu", top_k: int = 5) -> Dict[str, Any]:
        """
        Runs the full production RAG path (retrieve + generate) via HTTP or in-process router.
        Always returns: answer, documents, retrieval_timings, llm_ms, ttft_ms, total_ms, trace_id.
        """
        if self.is_server_online():
            url = f"{self.base_url}/api/v1/query/text"
            payload = json.dumps({
                "query": query,
                "language": language,
                "top_k": top_k,
                "evaluate": False
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    lat = data.get("latency", {}) or {}
                    return {
                        "answer": data.get("answer", ""),
                        "documents": data.get("sources", []),
                        "retrieval_timings": lat.get("retrieval") or {},
                        "llm_ms": lat.get("llm_ms") or 0.0,
                        "ttft_ms": lat.get("ttft_ms") or 0.0,
                        "total_ms": lat.get("total_ms") or 0.0,
                        "trace_id": data.get("trace_id"),
                    }
            except Exception as e:
                print(f"⚠️ API query failed ({e}), falling back to in-process router...")

        # In-process fallback: the very same pipeline.ask() the API uses
        if self.fallback_router:
            pipeline = self.fallback_router.get_pipeline(language)
            res = pipeline.ask(query, final_k=top_k)
            return {
                "answer": res.get("answer", ""),
                "documents": res.get("documents", []),
                "retrieval_timings": res.get("retrieval_timings", {}),
                "llm_ms": res.get("llm_ms", 0.0),
                "ttft_ms": res.get("ttft_ms", 0.0),
                "total_ms": res.get("total_ms", 0.0),
                "trace_id": res.get("trace_id"),
            }

        raise RuntimeError("Neither live RAG server nor in-process router is available.")
