"""
API Request Schemas
===================
Pydantic schemas for incoming client requests (text query, retrieve, evaluate, TTS, feedback).
"""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.models import ChatTurn

MAX_QUERY_CHARS = 2000
MAX_HISTORY_TURNS = 20
MAX_TTS_TEXT_CHARS = 5000

SortBy = Literal["rrf", "dense", "sparse", "rerank"]

_LANGUAGE_ALIASES = {
    "gu": "gu", "gujarati": "gu", "gu-in": "gu", "guj": "gu", "guj-gujr": "gu",
    "hi": "hi", "hindi": "hi", "hi-in": "hi", "hin": "hi", "hin-deva": "hi",
    "auto": "auto", "detect": "auto",
}
_TTS_LANGUAGE_ALIASES = {
    "gu": "gu", "gujarati": "gu", "gu-in": "gu",
    "hi": "hi", "hindi": "hi", "hi-in": "hi",
    "en": "en", "english": "en", "en-in": "en",
}


def normalize_request_language(value: Optional[str], allow_auto: bool = True, default: str = "gu") -> str:
    """Maps 'gu' | 'hi' | 'auto' and aliases ('Gujarati', 'hi-IN', 'hin_Deva', ...) to 'gu' | 'hi' | 'auto'."""
    if value is None or not str(value).strip():
        return default
    key = str(value).strip().lower().replace("_", "-")
    code = _LANGUAGE_ALIASES.get(key)
    if code is None or (code == "auto" and not allow_auto):
        allowed = "'gu', 'hi'" + (", 'auto'" if allow_auto else "")
        raise ValueError(f"Unsupported language '{value}'. Use {allowed}.")
    return code


def normalize_tts_language(value: Optional[str]) -> str:
    if value is None or not str(value).strip():
        return "gu"
    code = _TTS_LANGUAGE_ALIASES.get(str(value).strip().lower().replace("_", "-"))
    if code is None:
        raise ValueError(f"Unsupported TTS language '{value}'. Use 'gu', 'hi' or 'en'.")
    return code


class _LanguageRequest(BaseModel):
    language: str = Field(default="gu", description="Language for routing: 'gu', 'hi' or 'auto' (aliases like 'Gujarati', 'hi-IN' accepted).")

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, v):
        return normalize_request_language(v)


class TextQueryRequest(_LanguageRequest):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "query": "વાઇનની બોટલમાં કેટલા ઔંસ હોય છે",
            "language": "gu",
            "top_k": 5,
            "use_reranker": False,
            "sort_by": "rrf",
            "auto_detect_language": True,
            "evaluate": False,
            "voice_reply": False,
            "history": [],
        }]
    })

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS, description="The user query in Gujarati, Hindi, or English.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top context chunks to retrieve.")
    use_reranker: bool = Field(default=False, description="Whether to apply the secondary re-ranker stage.")
    sort_by: SortBy = Field(default="rrf", description="Sorting strategy for retrieved documents.")
    auto_detect_language: bool = Field(default=True, description="Route by the query's script (Gujarati / Devanagari) when it is unambiguous.")
    evaluate: bool = Field(default=False, description="Run DeepEval evaluation after generating the answer.")
    voice_reply: bool = Field(default=False, description="Synthesize the answer with Sarvam Bulbul v3.")
    query_id: Optional[int] = Field(default=None, description="Optional Golden query ID to force ground-truth matching.")
    history: Optional[List[ChatTurn]] = Field(
        default=None,
        max_length=MAX_HISTORY_TURNS,
        description=f"Previous conversation turns, oldest first (max {MAX_HISTORY_TURNS}; the server uses the most recent ones).",
    )


class TTSRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"text": "નમસ્તે! હું તમારી શું મદદ કરી શકું?", "language": "gu", "speaker": "shubh", "pace": 1.0}]
    })

    text: str = Field(..., min_length=1, max_length=MAX_TTS_TEXT_CHARS, description="Text to synthesize (long text is spoken up to RAG_TTS_MAX_CHARS).")
    language: str = Field(default="gu", description="'gu', 'hi' or 'en' (aliases 'gu-IN', 'hi-IN', 'en-IN' accepted).")
    speaker: Optional[str] = Field(default="shubh", min_length=2, max_length=32, pattern=r"^[a-z]+$", description="Bulbul v3 speaker (lowercase).")
    pace: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speech pace/speed.")

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, v):
        return normalize_tts_language(v)


class RetrieveRequest(_LanguageRequest):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"query": "મેનહટન પ્રોજેક્ટ શું હતો?", "language": "gu", "top_k": 5, "sort_by": "rrf"}]
    })

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS, description="Search query to retrieve passages for.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top retrieved passages.")
    use_reranker: bool = Field(default=False, description="Whether to apply secondary re-ranking.")
    sort_by: SortBy = Field(default="rrf", description="Sorting strategy.")
    auto_detect_language: bool = Field(default=True, description="Whether to automatically detect language script.")


class EvaluateRequest(_LanguageRequest):
    question: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS, description="The original question.")
    generated_answer: str = Field(..., min_length=1, max_length=8000, description="The answer generated by the system.")
    retrieved_contexts: List[str] = Field(default_factory=list, max_length=50, description="Context strings passed to the generator.")
    query_id: Optional[int] = Field(default=None, description="Optional Golden query ID.")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"trace_id": "0d4f5a43-9a3e-4c47-b7f3-1f0f6f1f9c55", "score": 1, "comment": "Correct and concise."}]
    })

    trace_id: Optional[str] = Field(default=None, max_length=64, description="LangSmith trace id returned with the answer (null when tracing is off).")
    score: Literal[0, 1] = Field(..., description="1 = thumbs up, 0 = thumbs down.")
    comment: Optional[str] = Field(default=None, max_length=2000, description="Optional free-text comment.")

    @field_validator("trace_id", mode="before")
    @classmethod
    def _validate_trace_id(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return str(uuid.UUID(str(v).strip()))
        except (ValueError, AttributeError, TypeError):
            raise ValueError("trace_id must be a UUID.")
