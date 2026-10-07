"""
API Security Utilities
======================
Small, dependency-free protection layer for the public API:

- API key auth (`X-API-Key` header / `api_key` query param for WebSockets).
  Disabled when `RAG_API_KEY` is unset (a loud warning is printed at startup).
- In-memory sliding-window rate limiter keyed by client IP (`RAG_RATE_LIMIT_PER_MIN`).
- CORS origin parsing (`RAG_CORS_ORIGINS`).
- Upload size limit (`RAG_MAX_UPLOAD_MB`) enforced on Content-Length and while streaming the body.
- Live voice WebSocket session limit (`RAG_WS_MAX_SECONDS`).

Settings are read from the environment on every call (cheap), so tests and operators
can change them without re-importing the app.

Client IPs: behind a reverse proxy, run uvicorn with
`--proxy-headers --forwarded-allow-ips=<proxy ip>` so `request.client.host` is the real
client address; otherwise every user shares the proxy's rate-limit bucket.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocket

API_KEY_HEADER_NAME = "X-API-Key"
WS_API_KEY_PARAM = "api_key"
WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_RATE_LIMITED = 4429

DEFAULT_CORS_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"
DEFAULT_RATE_LIMIT_PER_MIN = 30
DEFAULT_MAX_UPLOAD_MB = 10.0
DEFAULT_WS_MAX_SECONDS = 60.0
# Multipart framing + the small form fields sent next to the audio file.
MULTIPART_OVERHEAD_BYTES = 64 * 1024

ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4")
# Paths whose request body is size-limited by `UploadSizeLimitMiddleware`.
UPLOAD_PATHS = ("/api/v1/query/voice",)


# ------------------------------------------------------------
# Settings (read from env at call time)
# ------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ [security] Invalid {name}={raw!r}; using default {default}.")
        return default


def configured_api_key() -> Optional[str]:
    key = (os.environ.get("RAG_API_KEY") or "").strip().strip("\"'")
    return key or None


def auth_enabled() -> bool:
    return configured_api_key() is not None


def rate_limit_per_min() -> int:
    """Requests allowed per client IP per rolling minute (0 or negative disables the limiter)."""
    return int(_env_float("RAG_RATE_LIMIT_PER_MIN", DEFAULT_RATE_LIMIT_PER_MIN))


def max_upload_mb() -> float:
    return _env_float("RAG_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)


def max_upload_bytes() -> int:
    return int(max_upload_mb() * 1024 * 1024)


def ws_max_seconds() -> float:
    return max(1.0, _env_float("RAG_WS_MAX_SECONDS", DEFAULT_WS_MAX_SECONDS))


def cors_origins() -> List[str]:
    """Comma-separated `RAG_CORS_ORIGINS` (default: the local UI origins). `*` must be set explicitly."""
    raw = os.environ.get("RAG_CORS_ORIGINS")
    if raw is None or not raw.strip():
        raw = DEFAULT_CORS_ORIGINS
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return ["*"] if "*" in origins else origins


def log_security_config() -> None:
    """Prints the effective security settings once at startup."""
    if auth_enabled():
        print(f"🔐 [security] API key auth ON (header '{API_KEY_HEADER_NAME}', WebSocket query '{WS_API_KEY_PARAM}').")
    else:
        print(
            "⚠️ [security] RAG_API_KEY is not set: the API is OPEN. Anyone who can reach this server "
            "can spend your Groq / Sarvam credits. Set RAG_API_KEY before exposing it beyond localhost."
        )
    limit = rate_limit_per_min()
    print(
        f"🛡️ [security] rate limit={'off' if limit <= 0 else f'{limit}/min per IP'}, "
        f"max upload={max_upload_mb():g} MB, live voice max={ws_max_seconds():g}s, "
        f"CORS origins={cors_origins()}"
    )


# ------------------------------------------------------------
# API key
# ------------------------------------------------------------

def is_valid_api_key(provided: Optional[str]) -> bool:
    """Constant-time comparison against `RAG_API_KEY` (always True when auth is disabled)."""
    expected = configured_api_key()
    if expected is None:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.strip().encode("utf-8"), expected.encode("utf-8"))


_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="API key (required when the server sets RAG_API_KEY).",
)


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """FastAPI dependency: 401 unless the `X-API-Key` header matches `RAG_API_KEY`."""
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Send it in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ------------------------------------------------------------
# Rate limiting
# ------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window limiter (per process; use a shared store with >1 worker)."""

    def __init__(self, window_seconds: float = 60.0):
        self.window = float(window_seconds)
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key: str, limit: int, now: Optional[float] = None) -> Tuple[bool, int]:
        """Records one request for `key`. Returns (allowed, retry_after_seconds)."""
        if limit <= 0:
            return True, 0
        now = time.monotonic() if now is None else now
        cutoff = now - self.window
        with self._lock:
            self._calls += 1
            if self._calls % 1000 == 0:
                self._sweep(cutoff)
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                retry_after = max(1, int(math.ceil(q[0] + self.window - now)))
                return False, retry_after
            q.append(now)
            return True, 0

    def _sweep(self, cutoff: float) -> None:
        stale = [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]
        for k in stale:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._calls = 0


rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)


def client_ip(conn) -> str:
    client = getattr(conn, "client", None)
    host = getattr(client, "host", None) if client else None
    return host or "unknown"


def check_rate_limit(key: str) -> Tuple[bool, int]:
    return rate_limiter.hit(key, rate_limit_per_min())


async def rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 + Retry-After when the client IP exceeds `RAG_RATE_LIMIT_PER_MIN`."""
    allowed, retry_after = check_rate_limit(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({rate_limit_per_min()} requests/minute). Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


# Dependency bundles used by the routers.
PROTECTED = [Depends(require_api_key)]
PROTECTED_LIMITED = [Depends(require_api_key), Depends(rate_limit)]


# ------------------------------------------------------------
# WebSocket guard
# ------------------------------------------------------------

async def authorize_websocket(websocket: WebSocket) -> bool:
    """
    Accepts the socket, then validates `?api_key=` and the per-IP rate limit.
    On failure sends `{"type": "error", ...}` and closes with 4401 (auth) / 4429 (rate limit).
    The socket is accepted first because a close *before* accept reaches browsers as 1006,
    without the application close code.
    """
    await websocket.accept()
    if not is_valid_api_key(websocket.query_params.get(WS_API_KEY_PARAM)):
        await _ws_reject(websocket, WS_CLOSE_UNAUTHORIZED, "Missing or invalid API key (query param 'api_key').")
        return False
    allowed, retry_after = check_rate_limit(client_ip(websocket))
    if not allowed:
        await _ws_reject(websocket, WS_CLOSE_RATE_LIMITED, f"Rate limit exceeded. Retry in {retry_after}s.")
        return False
    return True


async def _ws_reject(websocket: WebSocket, code: int, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "message": message, "code": code})
    except Exception:
        pass
    try:
        await websocket.close(code=code, reason=message[:120])
    except Exception:
        pass


# ------------------------------------------------------------
# Upload size limit (ASGI middleware, runs before multipart parsing)
# ------------------------------------------------------------

def too_large_detail() -> str:
    return f"Uploaded file is too large (max {max_upload_mb():g} MB)."


class UploadSizeLimitMiddleware:
    """
    Rejects oversized bodies on upload endpoints *before* Starlette spools them to disk:
    - Content-Length above the limit -> immediate 413.
    - Chunked bodies are counted while streaming; going over raises HTTPException(413),
      which FastAPI re-raises from its body parser.
    The endpoint additionally re-checks the size of the file it actually reads.
    """

    def __init__(self, app: ASGIApp, paths: Tuple[str, ...] = UPLOAD_PATHS):
        self.app = app
        self.paths = tuple(paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        limit = max_upload_bytes() + MULTIPART_OVERHEAD_BYTES
        content_length = None
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                content_length = value.decode("latin-1").strip()
                break
        if content_length and content_length.isdigit() and int(content_length) > limit:
            await self._send_413(send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=too_large_detail())
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _send_413(send: Send) -> None:
        body = json.dumps({"detail": too_large_detail()}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
