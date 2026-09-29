"""
LangSmith Tracing Utilities
===========================
Centralized tracing integration for the Voice RAG system.
Supports async feedback logging, OpenAI LLM client wrapping, and run metadata.
"""

import os
import threading
from typing import Any, Dict, Iterable, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

try:
    import langsmith
    from langsmith import traceable as _ls_traceable
    from langsmith.run_helpers import get_current_run_tree
    from langsmith.wrappers import wrap_openai as _ls_wrap_openai
    _LANGSMITH_AVAILABLE = True
except ImportError:
    langsmith = None
    _LANGSMITH_AVAILABLE = False


def tracing_enabled() -> bool:
    flag = (os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or "").strip().lower()
    return _LANGSMITH_AVAILABLE and flag in ("true", "1", "yes") and bool(os.environ.get("LANGSMITH_API_KEY"))


def project_name() -> str:
    return (os.environ.get("LANGSMITH_PROJECT") or "Voice RAG System").strip().strip('"')


def traceable(*args, **kwargs):
    """`langsmith.traceable` when available, otherwise a transparent decorator."""
    if _LANGSMITH_AVAILABLE:
        return _ls_traceable(*args, **kwargs)
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda fn: fn


def current_run_id() -> Optional[str]:
    """Run id of the innermost active trace span (None when not tracing)."""
    if not tracing_enabled():
        return None
    run = get_current_run_tree()
    return str(run.id) if run else None


def current_trace_id() -> Optional[str]:
    """Root trace id of the active trace (None when not tracing)."""
    if not tracing_enabled():
        return None
    run = get_current_run_tree()
    return str(run.trace_id) if run else None


def add_run_metadata(**metadata: Any) -> None:
    """Attach metadata (language, model, latency, ...) to the active span."""
    if not tracing_enabled():
        return
    run = get_current_run_tree()
    if run is not None:
        run.metadata.update({k: v for k, v in metadata.items() if v is not None})


# ------------------------------------------------------------
# LLM clients (Groq through the OpenAI SDK so wrap_openai traces them)
# ------------------------------------------------------------

_client_lock = threading.Lock()
_groq_llm_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    """Shared, connection-pooled Groq client. Traced as `llm` runs when LangSmith is on."""
    global _groq_llm_client
    if _groq_llm_client is None:
        with _client_lock:
            if _groq_llm_client is None:
                client = OpenAI(
                    api_key=os.environ.get("GROQ_API_KEY", "").strip().strip('"'),
                    base_url=GROQ_BASE_URL,
                    max_retries=1,
                    timeout=30.0,
                )
                _groq_llm_client = _ls_wrap_openai(client) if tracing_enabled() else client
    return _groq_llm_client


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------

def as_langsmith_documents(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retriever output in the shape LangSmith renders as a document list."""
    out = []
    for rank, d in enumerate(documents, start=1):
        meta = {k: v for k, v in d.items() if k != "text"}
        meta.setdefault("rank", rank)
        out.append({"page_content": d.get("text", ""), "type": "Document", "metadata": meta})
    return out


# ------------------------------------------------------------
# Feedback (attach evaluation scores to production traces)
# ------------------------------------------------------------

_ls_client = None


def _get_ls_client():
    global _ls_client
    if _ls_client is None and tracing_enabled():
        _ls_client = langsmith.Client()
    return _ls_client


def log_eval_feedback(run_id: Optional[str], eval_result: Optional[Dict[str, Any]]) -> None:
    """Writes each numeric DeepEval score as LangSmith feedback on the request trace (in the background)."""
    if not run_id or not eval_result or not tracing_enabled():
        return
    threading.Thread(target=_send_feedback, args=(run_id, eval_result), daemon=True).start()


def _send_feedback(run_id: str, eval_result: Dict[str, Any]) -> None:
    scores = eval_result.get("scores") or {}
    client = _get_ls_client()
    if client is None:
        return
    comment = scores.get("reason") or eval_result.get("warning")
    for key, value in scores.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                client.create_feedback(
                    run_id,
                    key=key,
                    score=float(value),
                    comment=(comment[:2000] if comment and key == "overall_score" else None),
                    source_info={"evaluator": "deepeval", "is_golden": eval_result.get("is_golden", False)},
                )
            except Exception as e:
                print(f"⚠️ [LangSmith] Failed to log feedback '{key}': {e}")


def flush() -> None:
    """Blocks until pending traces are sent (call at shutdown / end of CLI runs)."""
    client = _get_ls_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
