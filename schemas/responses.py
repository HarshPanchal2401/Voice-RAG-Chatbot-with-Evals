"""
API Response Schemas
====================
Pydantic schemas for outgoing server responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from schemas.models import (
    SourceDocument,
    LatencyBreakdown,
    RetrievalTimings,
    EvaluationResult,
    LanguageInfo
)


class AvailableLanguagesResponse(BaseModel):
    languages: List[LanguageInfo] = Field(..., description="List of supported languages.")
    default_language: str = Field("gu", description="Default language code.")


class RAGResponse(BaseModel):
    status: str = Field("success", description="Response status.")
    mode: str = Field("text", description="Input mode: 'text' or 'voice'.")
    language: str = Field("gu", description="Language used for query and answer generation.")
    query: str = Field(..., description="Processed user query in text format.")
    transcription: Optional[str] = Field(None, description="Audio transcription (present if mode='voice').")
    answer: str = Field(..., description="Final AI generated answer in requested language.")
    sources: List[SourceDocument] = Field(..., description="List of top retrieved source documents.")
    latency: LatencyBreakdown = Field(..., description="Comprehensive stage-by-stage latency analysis in ms.")
    evaluation: Optional[EvaluationResult] = Field(None, description="DeepEval evaluation results and quality metrics.")
    trace_id: Optional[str] = Field(None, description="LangSmith trace id of this request (when tracing is enabled).")


class RetrievalResponse(BaseModel):
    status: str = Field("success", description="Response status.")
    query: str = Field(..., description="Query searched.")
    language: str = Field("gu", description="Language of retrieval.")
    top_k: int = Field(..., description="Number of results requested.")
    sources: List[SourceDocument] = Field(..., description="Retrieved passage chunks with ranks and scores.")
    retrieval_timings: RetrievalTimings = Field(..., description="Retrieval latency breakdown.")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health status.")
    device: str = Field(..., description="Active compute device (cuda / cpu).")
    loaded_languages: List[str] = Field(..., description="List of loaded language codes.")
    indexed_passages: int = Field(..., description="Total vectors indexed across all languages.")
    golden_dataset_records: int = Field(..., description="Total golden evaluation records loaded.")
    embedding_model: str = Field(..., description="Embedding model name.")
    llm_model: str = Field(..., description="Generation LLM model name.")
    language_details: Dict[str, Any] = Field(default={}, description="Per-language statistics.")
