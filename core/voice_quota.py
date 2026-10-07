"""
Voice quota (protects the Sarvam AI free-tier budget)
====================================================
Every request that calls Sarvam (speech-to-text and/or text-to-speech) costs one "voice use".
Each user gets `RAG_VOICE_LIMIT_PER_USER` uses in total (default 5, never resets).
Set `RAG_VOICE_LIMIT_WINDOW_HOURS` > 0 to make it a rolling window instead.

Who is "a user"? There are no accounts, so a user is identified two ways and BOTH are counted:
  - an anonymous browser id (header `X-Client-Id`, query `client_id` for WebSockets), and
  - the client IP (real visitor IP from `X-Forwarded-For` when `RAG_TRUST_PROXY=1`, e.g. on
    Hugging Face Spaces / behind Cloudflare; otherwise the socket address).
A request is blocked when EITHER identity is over the limit, so clearing browser storage or
switching browsers on the same network does not reset the quota.

Optional global cap: `RAG_VOICE_GLOBAL_DAILY_LIMIT` (0 = off) limits all users together per day.
Usage is kept in memory and persisted to `RAG_VOICE_QUOTA_FILE` (default Data/voice_quota.json)
so restarts do not reset it (on Spaces the disk is ephemeral, so a rebuild does reset it).
"""

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_GLOBAL_KEY = "global:all"


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def limit_per_user() -> int:
    return _int_env("RAG_VOICE_LIMIT_PER_USER", 5)


def window_seconds() -> Optional[int]:
    """None = lifetime limit (never resets). Default: lifetime."""
    hours = _int_env("RAG_VOICE_LIMIT_WINDOW_HOURS", 0)
    return hours * 3600 if hours > 0 else None


def global_daily_limit() -> int:
    return _int_env("RAG_VOICE_GLOBAL_DAILY_LIMIT", 0)


def trust_proxy() -> bool:
    return os.environ.get("RAG_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")


def real_client_ip(conn) -> str:
    """Visitor IP. Uses the left-most X-Forwarded-For entry only when RAG_TRUST_PROXY is on."""
    if trust_proxy():
        fwd = conn.headers.get("x-forwarded-for") or conn.headers.get("x-real-ip") or ""
        first = fwd.split(",")[0].strip()
        if first:
            return first
    client = getattr(conn, "client", None)
    return (getattr(client, "host", None) if client else None) or "unknown"


def client_id(conn) -> Optional[str]:
    raw = conn.headers.get("x-client-id") or conn.query_params.get("client_id") or ""
    raw = raw.strip()
    return raw if _CLIENT_ID_RE.match(raw) else None


def identities(conn) -> List[str]:
    ids = ["ip:" + real_client_ip(conn)]
    cid = client_id(conn)
    if cid:
        ids.append("cid:" + cid)
    return ids


class VoiceQuota:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.environ.get("RAG_VOICE_QUOTA_FILE") or (_ROOT / "Data" / "voice_quota.json"))
        self._lock = threading.Lock()
        self._usage: Dict[str, List[float]] = {}
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._usage = {k: [float(t) for t in v] for k, v in data.items() if isinstance(v, list)}
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ [voice_quota] Could not read {self.path}: {e}")

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._usage), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"⚠️ [voice_quota] Could not save {self.path}: {e}")

    # ---------- core ----------
    def _prune(self, now: float) -> None:
        win = window_seconds()
        day_cut = now - 86400
        for k in list(self._usage):
            if k != _GLOBAL_KEY and win is None:
                continue  # lifetime limit: keep every use forever
            cut = day_cut if k == _GLOBAL_KEY else now - win
            kept = [t for t in self._usage[k] if t > cut]
            if kept:
                self._usage[k] = kept
            else:
                del self._usage[k]

    def _status(self, keys: List[str], now: float) -> Dict[str, object]:
        limit = limit_per_user()
        used = max((len(self._usage.get(k, [])) for k in keys), default=0)
        oldest = [self._usage[k][0] for k in keys if self._usage.get(k)]
        win = window_seconds()
        reset_in = int(max(0, min(oldest) + win - now)) if (win and oldest and used >= limit) else 0
        g_limit = global_daily_limit()
        g_used = len(self._usage.get(_GLOBAL_KEY, []))
        global_blocked = g_limit > 0 and g_used >= g_limit
        if global_blocked and self._usage.get(_GLOBAL_KEY):
            reset_in = max(reset_in, int(max(0, self._usage[_GLOBAL_KEY][0] + 86400 - now)))
        return {
            "enabled": limit > 0,
            "limit": limit,
            "used": min(used, limit) if limit > 0 else used,
            "remaining": max(0, limit - used) if limit > 0 else None,
            "window_hours": (window_seconds() // 3600) if window_seconds() else None,
            "lifetime": window_seconds() is None,
            "reset_in_seconds": reset_in,
            "global_exhausted": global_blocked,
            "allowed": (limit <= 0 or used < limit) and not global_blocked,
        }

    def status(self, conn) -> Dict[str, object]:
        with self._lock:
            now = time.time()
            self._prune(now)
            return self._status(identities(conn), now)

    def try_consume(self, conn) -> Tuple[bool, Dict[str, object]]:
        """Atomically check and record one voice use. Returns (allowed, status_after)."""
        with self._lock:
            now = time.time()
            self._prune(now)
            keys = identities(conn)
            st = self._status(keys, now)
            if not st["allowed"]:
                return False, st
            if limit_per_user() > 0:
                for k in keys:
                    self._usage.setdefault(k, []).append(now)
            if global_daily_limit() > 0:
                self._usage.setdefault(_GLOBAL_KEY, []).append(now)
            self._save()
            return True, self._status(keys, now)


def limit_message(st: Dict[str, object]) -> str:
    if st.get("global_exhausted"):
        base = "The daily voice budget for this demo is used up."
    else:
        base = (f"You have used all {st.get('limit')} free voice uses." if st.get("lifetime")
                else f"Voice limit reached ({st.get('limit')} voice uses per {st.get('window_hours')} h).")
    secs = int(st.get("reset_in_seconds") or 0)
    if secs:
        h, m = divmod(secs // 60, 60)
        base += f" Try again in {h} h {m} min." if h else f" Try again in {max(m, 1)} min."
    return base + " You can still type your questions."


voice_quota = VoiceQuota()
