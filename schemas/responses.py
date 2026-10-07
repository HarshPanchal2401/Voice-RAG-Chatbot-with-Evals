"""
API Response Schemas
====================
Pydantic schemas for outgoing server responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from schemas.models import (
    SourceDocument,
    LatencyBreakdown,
    RetrievalTimings,
    EvaluationResult,
    LanguageInfo,
    SampleQuery,
)


class AvailableLanguagesResponse(BaseModel):
    languages: List[LanguageInfo] = Field(..., description="List of supported languages.")
    default_language: str = Field("gu", description="Default language code.")


class RAGResponse(BaseModel):
    status: str = Field("success", description="Response status.")
    mode: str = Field("text", description="Input mode: 'text' or 'voice'.")
    language: str = Field("gu", description="Index (corpus) language used for retrieval: 'gu' | 'hi'.")
    answer_language: Optional[str] = Field(None, description="Language of the generated answer: 'gu' | 'hi' | 'en'.")
    query: str = Field(..., description="Processed user query in text format.")
    retrieval_query: Optional[str] = Field(None, description="Standalone query actually used for retrieval (condensed from history).")
    transcription: Optional[str] = Field(None, description="Audio transcription (present if mode='voice').")
    answer: str = Field(..., description="Final AI generated answer.")
    no_answer: bool = Field(False, description="True when the context did not contain the answer.")
    sources: List[SourceDocument] = Field(..., description="List of top retrieved source documents.")
    latency: LatencyBreakdown = Field(..., description="Comprehensive stage-by-stage latency analysis in ms.")
    evaluation: Optional[EvaluationResult] = Field(None, description="DeepEval evaluation results and quality metrics.")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded MP3 audio of the answer (when voice_reply is enabled).")
    trace_id: Optional[str] = Field(None, description="LangSmith trace id of this request (when tracing is enabled).")


class TTSResponse(BaseModel):
    status: str = Field("success", description="TTS response status.")
    language: str = Field(..., description="TTS language code ('gu' | 'hi' | 'en').")
    speaker: str = Field("shubh", description="Voice speaker used.")
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio.")
    tts_ms: float = Field(..., description="TTS synthesis latency in ms.")
    chunks: Optional[int] = Field(None, description="Number of sentence chunks synthesized.")
    spoken_chars: Optional[int] = Field(None, description="Characters actually spoken (capped by RAG_TTS_MAX_CHARS).")


class RetrievalResponse(BaseModel):
    status: str = Field("success", description="Response status.")
    query: str = Field(..., description="Query searched.")
    language: str = Field("gu", description="Language of retrieval.")
    top_k: int = Field(..., description="Number of results requested.")
    sources: List[SourceDocument] = Field(..., description="Retrieved passage chunks with ranks and scores.")
    retrieval_timings: RetrievalTimings = Field(..., description="Retrieval latency breakdown.")


class SampleQueriesResponse(BaseModel):
    status: str = Field("success", description="Response status.")
    language: str = Field(..., description="Language of the sampled queries.")
    count: int = Field(..., description="Number of queries returned.")
    queries: List[SampleQuery] = Field(..., description="Sampled golden questions (no ground-truth answers).")


class FeedbackResponse(BaseModel):
    status: str = Field("ok", description="Always 'ok' when the feedback was accepted.")
    logged: bool = Field(False, description="True when the feedback was written to LangSmith.")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health status.")
    device: str = Field(..., description="Active compute device (cuda / cpu).")
    loaded_languages: List[str] = Field(..., description="List of loaded language codes.")
    indexed_passages: int = Field(..., description="Total vectors indexed across all languages.")
    golden_dataset_records: int = Field(..., description="Total golden evaluation records loaded.")
    embedding_model: str = Field(..., description="Embedding model name.")
    llm_model: str = Field(..., description="Generation LLM model name.")
    reranker_backend: str = Field("llm", description="Re-ranker backend: 'llm' | 'cross_encoder' | 'lexical'.")
    auth_required: bool = Field(False, description="True when /api/* routes need the X-API-Key header.")
    tracing_enabled: bool = Field(False, description="True when LangSmith tracing (and feedback logging) is on.")
    language_details: Dict[str, Any] = Field(default_factory=dict, description="Per-language statistics.")
