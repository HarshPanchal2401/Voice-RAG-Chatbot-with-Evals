"""
Voice Service — Audio Transcription & Speech-to-Text
====================================================
Unified STT service supporting Sarvam AI (saaras:v3) and Groq Whisper (whisper-large-v3)
for Gujarati (gu-IN) and Hindi (hi-IN).
"""

import io
import os
import time
import requests
from typing import Tuple, Optional, Dict, Any
from groq import Groq
from dotenv import load_dotenv

from core.tracing import traceable, add_run_metadata
from pipeline.generator import detect_script_language

load_dotenv()


def _stt_inputs(inputs: dict) -> dict:
    audio = inputs.get("audio_bytes") or b""
    return {
        "filename": inputs.get("filename"),
        "language": inputs.get("language"),
        "audio_kb": round(len(audio) / 1024, 1),
    }


class VoiceService:
    """
    Unified Multilingual Speech-to-Text (STT) Service.
    Routes Gujarati (gu-IN) and Hindi (hi-IN) audio streams to Sarvam AI or Groq Whisper.
    """

    LANGUAGE_MAP = {
        "gu": {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"},
        "gujarati": {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"},
        "gu-in": {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"},
        "hi": {"sarvam": "hi-IN", "whisper": "hi", "name": "Hindi"},
        "hindi": {"sarvam": "hi-IN", "whisper": "hi", "name": "Hindi"},
        "hi-in": {"sarvam": "hi-IN", "whisper": "hi", "name": "Hindi"},
    }

    def __init__(self, groq_client: Optional[Groq] = None):
        self.groq_client = groq_client or Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.sarvam_api_key = os.environ.get("SARVAM_API_KEY", "").strip()
        # Keep-alive session avoids repeated TLS handshakes
        self._http = requests.Session()

    def _resolve_lang(self, lang: str) -> dict:
        key = (lang or "gu").strip().lower()
        return self.LANGUAGE_MAP.get(key, {"sarvam": "gu-IN", "whisper": "gu", "name": "Gujarati"})

    @traceable(run_type="tool", name="speech_to_text", process_inputs=_stt_inputs)
    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "input_audio.wav",
        language: str = "gu",
        prefer_sarvam: bool = True
    ) -> Tuple[str, float]:
        """
        Transcribes audio bytes to text in requested language.
        Returns: (transcribed_text, stt_latency_ms)
        """
        t0 = time.perf_counter()
        lang_info = self._resolve_lang(language)

        # 1. Try Sarvam AI if available and preferred
        if prefer_sarvam and self.sarvam_api_key:
            try:
                text = self._transcribe_sarvam(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    language_code=lang_info["sarvam"]
                )
                if text and text.strip():
                    latency_ms = (time.perf_counter() - t0) * 1000
                    add_run_metadata(provider="sarvam", model="saaras:v3", stt_ms=round(latency_ms, 2))
                    return text.strip(), latency_ms
            except Exception as e:
                print(f"⚠️ [VoiceService] Sarvam AI STT ({lang_info['sarvam']}) failed ({e}), falling back to Groq Whisper...")

        # 2. Fallback to Groq Whisper
        text = self._transcribe_groq(
            audio_bytes=audio_bytes,
            filename=filename,
            whisper_lang=lang_info["whisper"]
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        add_run_metadata(provider="groq-whisper", stt_ms=round(latency_ms, 2))
        return text.strip(), latency_ms

    def _transcribe_sarvam(self, audio_bytes: bytes, filename: str, language_code: str = "gu-IN") -> str:
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {
            "api-subscription-key": self.sarvam_api_key
        }

        ext = os.path.splitext(filename)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
            ".flac": "audio/flac",
        }
        content_type = mime_map.get(ext, "audio/wav")

        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data = {
            "language_code": language_code,
            "model": "saaras:v3",
        }
        resp = self._http.post(url, headers=headers, files=files, data=data, timeout=15)
        resp.raise_for_status()
        res_json = resp.json()
        return res_json.get("transcript", "")

    def _transcribe_groq(self, audio_bytes: bytes, filename: str, whisper_lang: str = "gu") -> str:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4"]:
            ext = ".wav"
            filename = f"audio{ext}"

        audio_file = (filename, io.BytesIO(audio_bytes))

        try:
            transcription = self.groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language=whisper_lang,
                response_format="json"
            )
            text = getattr(transcription, "text", "")
            if text and text.strip():
                return text.strip()
        except Exception as e:
            print(f"⚠️ Groq whisper-large-v3-turbo ({whisper_lang}) error ({e}), retrying whisper-large-v3...")

        # Fallback to standard whisper-large-v3
        transcription = self.groq_client.audio.transcriptions.create(
            file=(filename, io.BytesIO(audio_bytes)),
            model="whisper-large-v3",
            language=whisper_lang,
            response_format="json"
        )
        return getattr(transcription, "text", "").strip()

    @staticmethod
    def _clean_text_for_speech(text: str) -> str:
        """Removes markdown tags, headers, and reference brackets for smooth voice synthesis."""
        import re
        if not text:
            return ""
        # Remove citation references like [સંદર્ભ 1], [संदर्भ 2], [Doc 1], [1]
        text = re.sub(r'\[(?:સંદર્ભ|संदर्भ|Doc|Chunk|\d+)[^\]]*\]', '', text)
        # Remove markdown bold/italics
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        # Remove markdown headers
        text = re.sub(r'#+\s*', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @traceable(run_type="tool", name="text_to_speech")
    def generate_tts_audio(
        self,
        text: str,
        language: str = "gu",
        speaker: str = "shubh",
        pace: float = 1.0,
        speech_sample_rate: int = 22050,
        model: str = "bulbul:v3"
    ) -> Tuple[Optional[bytes], float]:
        """
        Synthesizes speech audio from text using Sarvam AI bulbul:v3.
        Supports Gujarati ('gu-IN') and Hindi ('hi-IN').
        Returns: (audio_mp3_bytes, tts_latency_ms)
        """
        if not text or not text.strip() or not self.sarvam_api_key:
            return None, 0.0

        t0 = time.perf_counter()
        lang_info = self._resolve_lang(language)
        lang_code = lang_info["sarvam"]  # 'gu-IN' or 'hi-IN'

        clean_text = self._clean_text_for_speech(text)
        if not clean_text:
            return None, 0.0

        # Limit to 500 characters to keep latency fast for voice conversation
        if len(clean_text) > 500:
            clean_text = clean_text[:500].rsplit(".", 1)[0] + "."

        try:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=self.sarvam_api_key)
            chunks = list(client.text_to_speech.convert_stream(
                text=clean_text,
                language_code=lang_code,
                speaker=speaker,
                model=model,
                pace=pace,
                speech_sample_rate=speech_sample_rate,
                output_audio_codec="mp3"
            ))
            audio_bytes = b"".join(chunks)
            tts_ms = (time.perf_counter() - t0) * 1000
            add_run_metadata(tts_model=model, tts_lang=lang_code, tts_ms=round(tts_ms, 2))
            return audio_bytes, tts_ms
        except Exception as e:
            print(f"⚠️ [VoiceService] Sarvam TTS error ({lang_code}, {model}): {e}")
            return None, 0.0

