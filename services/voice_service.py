"""
Voice Service — Speech-to-Text & Text-to-Speech
===============================================
- STT: Sarvam AI `saaras:v3` (REST) with Groq Whisper fallback, for Gujarati / Hindi (and English).
  Auto language detection: Sarvam REST accepts `language_code="unknown"` and returns the detected
  BCP-47 `language_code`; Whisper is called without `language` so it detects it itself.
- TTS: Sarvam AI `bulbul:v3`. Text is cleaned (markdown, URLs, `[Source N]` citation tags), split on
  sentence boundaries (`. ? ! । ॥` and newlines), capped at `RAG_TTS_MAX_CHARS` (default 1500) at a
  sentence boundary, grouped into requests of at most `RAG_TTS_CHUNK_CHARS` (default 500, always below
  the SDK's per-request limit: 2500 chars for `convert` / 3500 for `convert_stream` with bulbul:v3),
  synthesized in parallel and concatenated as MP3.
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from core.tracing import traceable, add_run_metadata

load_dotenv()

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL = "saaras:v3"
SARVAM_STT_AUTO_CODE = "unknown"        # REST auto-detect value (SpeechToTextLanguage literal)
SARVAM_REALTIME_AUTO_CODE = "auto"      # realtime WebSocket auto-detect value
SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_TTS_REQUEST_LIMIT = 2500         # bulbul:v3 `convert` limit (convert_stream allows 3500)
DEFAULT_SPEAKER = "shubh"

AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def tts_max_chars() -> int:
    """Total characters spoken per answer (cut at a sentence boundary)."""
    return max(50, _env_int("RAG_TTS_MAX_CHARS", 1500))


def tts_chunk_chars() -> int:
    """Characters per Sarvam TTS request (kept below the SDK per-request limit)."""
    return max(80, min(_env_int("RAG_TTS_CHUNK_CHARS", 500), SARVAM_TTS_REQUEST_LIMIT))


# ------------------------------------------------------------
# Language helpers
# ------------------------------------------------------------

LANGUAGE_MAP: Dict[str, Dict[str, str]] = {
    "gu": {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"},
    "gujarati": {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"},
    "gu-in": {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"},
    "hi": {"sarvam": "hi-IN", "whisper": "hi", "name": "Hindi"},
    "hindi": {"sarvam": "hi-IN", "whisper": "hi", "name": "Hindi"},
    "hi-in": {"sarvam": "hi-IN", "whisper": "hi", "name": "Hindi"},
    "en": {"sarvam": "en-IN", "whisper": "en", "name": "English"},
    "english": {"sarvam": "en-IN", "whisper": "en", "name": "English"},
    "en-in": {"sarvam": "en-IN", "whisper": "en", "name": "English"},
}


def short_lang_code(code: Optional[str]) -> Optional[str]:
    """'gu-IN' / 'gujarati' -> 'gu', 'hi-IN' -> 'hi', 'en-IN' / 'english' -> 'en', anything else -> None."""
    if not code:
        return None
    key = str(code).strip().lower().replace("_", "-")
    if key in LANGUAGE_MAP:
        return LANGUAGE_MAP[key]["whisper"]
    base = key.split("-")[0]
    return base if base in ("gu", "hi", "en") else None


def script_language(text: str) -> Optional[str]:
    """'gu' / 'hi' from the dominant Indic script, 'en' for Latin-only text, else None."""
    if not text:
        return None
    gu = sum(1 for ch in text if "઀" <= ch <= "૿")
    hi = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    if gu > hi and gu > 0:
        return "gu"
    if hi > gu and hi > 0:
        return "hi"
    latin = sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    return "en" if latin >= 3 else None


# ------------------------------------------------------------
# Text cleaning & sentence chunking for TTS
# ------------------------------------------------------------

_CITATION_RE = re.compile(
    r"\[(?:\s*(?:source|sources|doc|chunk|ref|સંદર્ભ|સ્ત્રોત|स्रोत|संदर्भ)\s*)?[\d\s,;–-]+\]"
    r"|\[(?:source|sources|doc|chunk|ref|સંદર્ભ|સ્ત્રોત|स्रोत|संदर्भ)[^\]]{0,20}\]",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!।॥])\s+|\s*\n+\s*")


def clean_text_for_speech(text: str) -> str:
    """Removes citation tags, markdown and URLs; keeps line breaks (they are sentence boundaries)."""
    if not text:
        return ""
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{2}([^_\n]+)_{2}", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)          # headers
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d{1,2}[.)])\s+", "", text)  # list markers
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" +([.?!।॥,])", r"\1", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


_split_impl = None


def split_sentences(text: str) -> List[str]:
    """Sentence split on `. ? ! । ॥` and newlines (uses `pipeline.generator.split_sentences` when present)."""
    global _split_impl
    if _split_impl is None:
        try:
            from pipeline.generator import split_sentences as _pipeline_split  # type: ignore
            _split_impl = _pipeline_split
        except Exception:
            _split_impl = _local_split_sentences
    try:
        parts = _split_impl(text)
    except Exception:
        parts = _local_split_sentences(text)
    return [p.strip() for p in parts if p and p.strip()]


def _local_split_sentences(text: str) -> List[str]:
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "") if p and p.strip()]


def split_long_text(sentence: str, limit: int) -> List[str]:
    """Splits one over-long sentence at word boundaries into pieces of at most `limit` chars."""
    pieces, current = [], ""
    for word in sentence.split():
        if len(word) > limit:  # pathological token without spaces
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i:i + limit] for i in range(0, len(word), limit))
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def cap_sentences(sentences: List[str], max_total: int, chunk_limit: int) -> List[str]:
    """Keeps whole sentences while the total stays <= max_total (the first one is cut at a word boundary)."""
    kept: List[str] = []
    total = 0
    for sent in sentences:
        pieces = split_long_text(sent, chunk_limit) if len(sent) > chunk_limit else [sent]
        for piece in pieces:
            extra = len(piece) + (1 if kept else 0)
            if total + extra > max_total:
                if not kept:
                    kept.append(split_long_text(piece, max_total)[0])
                return kept
            kept.append(piece)
            total += extra
    return kept


def group_sentences(sentences: List[str], chunk_limit: int) -> List[str]:
    """Packs consecutive sentences into request-sized chunks (<= chunk_limit chars each)."""
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        if current and len(candidate) > chunk_limit:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def prepare_tts_chunks(text: str, max_total: Optional[int] = None, chunk_limit: Optional[int] = None) -> List[str]:
    """clean -> sentence split -> cap total at a sentence boundary -> group into request chunks."""
    max_total = max_total or tts_max_chars()
    chunk_limit = chunk_limit or tts_chunk_chars()
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return []
    sentences = cap_sentences(split_sentences(cleaned), max_total, chunk_limit)
    return group_sentences(sentences, chunk_limit)


def _stt_inputs(inputs: dict) -> dict:
    audio = inputs.get("audio_bytes") or b""
    return {
        "filename": inputs.get("filename"),
        "language": inputs.get("language"),
        "auto_detect": inputs.get("auto_detect"),
        "audio_kb": round(len(audio) / 1024, 1),
    }


def _tts_inputs(inputs: dict) -> dict:
    text = inputs.get("text") or ""
    return {"language": inputs.get("language"), "speaker": inputs.get("speaker"), "chars": len(text), "text": text[:300]}


def _tts_outputs(outputs: Any) -> Any:
    if isinstance(outputs, tuple) and outputs:
        audio = outputs[0] or b""
        return {"audio_kb": round(len(audio) / 1024, 1), "tts_ms": outputs[1] if len(outputs) > 1 else None}
    if isinstance(outputs, dict):
        return {k: v for k, v in outputs.items() if k != "audio"} | {"audio_kb": round(len(outputs.get("audio") or b"") / 1024, 1)}
    return outputs


# ------------------------------------------------------------
# Service
# ------------------------------------------------------------

class VoiceService:
    """Multilingual STT + TTS service (Sarvam AI first, Groq Whisper as STT fallback)."""

    LANGUAGE_MAP = LANGUAGE_MAP

    def __init__(self, groq_client: Optional[Any] = None, max_tts_workers: Optional[int] = None):
        self._groq_client = groq_client
        self.sarvam_api_key = os.environ.get("SARVAM_API_KEY", "").strip().strip('"')
        # Keep-alive session avoids repeated TLS handshakes for REST STT.
        self._http = requests.Session()
        self._sarvam_client = None
        self._client_lock = threading.Lock()
        workers = max_tts_workers or max(1, _env_int("RAG_TTS_WORKERS", 4))
        # Shared pool for TTS chunk requests (also used by the streaming routes).
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts")

    # ---------------- clients ----------------

    @property
    def groq_client(self):
        if self._groq_client is None:
            api_key = os.environ.get("GROQ_API_KEY", "").strip().strip('"')
            if api_key:
                from groq import Groq
                with self._client_lock:
                    if self._groq_client is None:
                        self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    @property
    def sarvam_client(self):
        """One shared SarvamAI client (httpx connection pool, thread-safe)."""
        if self._sarvam_client is None and self.sarvam_api_key:
            with self._client_lock:
                if self._sarvam_client is None:
                    from sarvamai import SarvamAI
                    self._sarvam_client = SarvamAI(api_subscription_key=self.sarvam_api_key, timeout=30.0)
        return self._sarvam_client

    def _resolve_lang(self, lang: Optional[str]) -> dict:
        key = (lang or "gu").strip().lower().replace("_", "-")
        return self.LANGUAGE_MAP.get(key, self.LANGUAGE_MAP["gu"])

    # ---------------- STT ----------------

    @traceable(run_type="tool", name="speech_to_text", process_inputs=_stt_inputs)
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "input_audio.wav",
        language: str = "gu",
        auto_detect: bool = False,
        prefer_sarvam: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribes audio once.
        Returns {"text", "stt_ms", "language" ('gu'|'hi'|'en'|None, detected or requested),
                 "language_code" (provider BCP-47 / name), "provider"}.
        With `auto_detect=True` (or language='auto') the provider detects the spoken language, so Hindi
        speech is not forced into Gujarati script when the UI language is Gujarati.
        """
        t0 = time.perf_counter()
        auto = auto_detect or (language or "").strip().lower() == "auto"
        lang_info = self._resolve_lang(None if (language or "").strip().lower() == "auto" else language)
        requested_short = lang_info["whisper"]

        if prefer_sarvam and self.sarvam_api_key:
            try:
                text, detected_code = self._transcribe_sarvam(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    language_code=SARVAM_STT_AUTO_CODE if auto else lang_info["sarvam"],
                )
                if text and text.strip():
                    stt_ms = (time.perf_counter() - t0) * 1000
                    detected = short_lang_code(detected_code) if auto else requested_short
                    if auto and detected is None:
                        detected = script_language(text)
                    add_run_metadata(provider="sarvam", model=SARVAM_STT_MODEL, stt_ms=round(stt_ms, 2),
                                     detected_language=detected_code)
                    return {"text": text.strip(), "stt_ms": stt_ms, "language": detected,
                            "language_code": detected_code, "provider": "sarvam"}
            except Exception as e:
                print(f"⚠️ [VoiceService] Sarvam STT failed ({e}); falling back to Groq Whisper...")

        text, whisper_lang = self._transcribe_groq(
            audio_bytes=audio_bytes,
            filename=filename,
            whisper_lang=None if auto else requested_short,
        )
        stt_ms = (time.perf_counter() - t0) * 1000
        detected = (short_lang_code(whisper_lang) or script_language(text)) if auto else requested_short
        add_run_metadata(provider="groq-whisper", stt_ms=round(stt_ms, 2), detected_language=whisper_lang)
        return {"text": (text or "").strip(), "stt_ms": stt_ms, "language": detected,
                "language_code": whisper_lang, "provider": "groq-whisper"}

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "input_audio.wav",
        language: str = "gu",
        prefer_sarvam: bool = True,
        auto_detect: bool = False,
    ) -> Tuple[str, float]:
        """Backward-compatible wrapper: returns (text, stt_ms)."""
        res = self.transcribe(audio_bytes, filename=filename, language=language,
                              auto_detect=auto_detect, prefer_sarvam=prefer_sarvam)
        return res["text"], res["stt_ms"]

    def _transcribe_sarvam(self, audio_bytes: bytes, filename: str, language_code: str = "gu-IN") -> Tuple[str, Optional[str]]:
        ext = os.path.splitext(filename or "")[1].lower()
        files = {"file": (filename or "audio.wav", audio_bytes, AUDIO_MIME_TYPES.get(ext, "audio/wav"))}
        data = {"language_code": language_code, "model": SARVAM_STT_MODEL}
        resp = self._http.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": self.sarvam_api_key},
            files=files,
            data=data,
            timeout=30,
        )
        resp.raise_for_status()
        res_json = resp.json()
        return res_json.get("transcript", "") or "", res_json.get("language_code")

    def _transcribe_groq(self, audio_bytes: bytes, filename: str, whisper_lang: Optional[str] = "gu") -> Tuple[str, Optional[str]]:
        """Groq Whisper; `whisper_lang=None` lets Whisper detect the language (verbose_json reports it)."""
        client = self.groq_client
        if client is None:
            raise RuntimeError("No STT provider available (set SARVAM_API_KEY or GROQ_API_KEY).")
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in AUDIO_MIME_TYPES:
            ext = ".wav"
            filename = f"audio{ext}"

        def _call(model: str):
            kwargs: Dict[str, Any] = {
                "file": (filename, io.BytesIO(audio_bytes)),
                "model": model,
                "response_format": "verbose_json" if whisper_lang is None else "json",
            }
            if whisper_lang:
                kwargs["language"] = whisper_lang
            res = client.audio.transcriptions.create(**kwargs)
            text = (getattr(res, "text", "") or "").strip()
            detected = getattr(res, "language", None)
            if detected is None:
                detected = (getattr(res, "model_extra", None) or {}).get("language")
            return text, (detected or whisper_lang)

        try:
            text, detected = _call("whisper-large-v3-turbo")
            if text:
                return text, detected
        except Exception as e:
            print(f"⚠️ Groq whisper-large-v3-turbo error ({e}), retrying whisper-large-v3...")
        return _call("whisper-large-v3")

    # ---------------- TTS ----------------

    @staticmethod
    def _clean_text_for_speech(text: str) -> str:
        """Backward-compatible alias of the module-level cleaner."""
        return clean_text_for_speech(text)

    def synthesize_chunk(
        self,
        text: str,
        language: str = "gu",
        speaker: str = DEFAULT_SPEAKER,
        pace: float = 1.0,
        speech_sample_rate: int = 22050,
        model: str = SARVAM_TTS_MODEL,
    ) -> Tuple[Optional[bytes], float]:
        """One Sarvam TTS request for an already cleaned, request-sized chunk. Returns (mp3_bytes|None, ms)."""
        text = (text or "").strip()
        client = self.sarvam_client
        if not text or client is None:
            return None, 0.0
        if len(text) > SARVAM_TTS_REQUEST_LIMIT:
            text = split_long_text(text, SARVAM_TTS_REQUEST_LIMIT)[0]
        lang_code = self._resolve_lang(language)["sarvam"]
        t0 = time.perf_counter()
        try:
            audio = b"".join(client.text_to_speech.convert_stream(
                text=text,
                language_code=lang_code,
                speaker=speaker or DEFAULT_SPEAKER,
                model=model,
                pace=pace or 1.0,
                speech_sample_rate=speech_sample_rate,
                output_audio_codec="mp3",
            ))
            return (audio or None), (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"⚠️ [VoiceService] Sarvam TTS error ({lang_code}, {len(text)} chars): {e}")
            return None, (time.perf_counter() - t0) * 1000

    @traceable(run_type="tool", name="text_to_speech", process_inputs=_tts_inputs, process_outputs=_tts_outputs)
    def generate_tts(
        self,
        text: str,
        language: str = "gu",
        speaker: str = DEFAULT_SPEAKER,
        pace: float = 1.0,
        speech_sample_rate: int = 22050,
        model: str = SARVAM_TTS_MODEL,
    ) -> Dict[str, Any]:
        """Full answer TTS: sentence chunks synthesized in parallel, MP3 bytes concatenated in order."""
        t0 = time.perf_counter()
        chunks = prepare_tts_chunks(text)
        if not chunks or self.sarvam_client is None:
            return {"audio": None, "tts_ms": 0.0, "chunks": 0, "spoken_chars": 0}
        results = list(self.executor.map(
            lambda c: self.synthesize_chunk(c, language, speaker, pace, speech_sample_rate, model), chunks
        ))
        audio = b"".join(a for a, _ in results if a)
        tts_ms = (time.perf_counter() - t0) * 1000
        lang_code = self._resolve_lang(language)["sarvam"]
        add_run_metadata(tts_model=model, tts_lang=lang_code, tts_ms=round(tts_ms, 2), tts_chunks=len(chunks))
        return {
            "audio": audio or None,
            "tts_ms": tts_ms,
            "chunks": len(chunks),
            "spoken_chars": sum(len(c) for c in chunks),
        }

    def generate_tts_audio(
        self,
        text: str,
        language: str = "gu",
        speaker: str = DEFAULT_SPEAKER,
        pace: float = 1.0,
        speech_sample_rate: int = 22050,
        model: str = SARVAM_TTS_MODEL,
    ) -> Tuple[Optional[bytes], float]:
        """Backward-compatible: returns (mp3_bytes|None, tts_ms)."""
        res = self.generate_tts(text, language, speaker, pace, speech_sample_rate, model)
        return res["audio"], res["tts_ms"]

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._http.close()
        except Exception:
            pass
