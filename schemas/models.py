"""
Domain Data Models & Response Components
========================================
Pydantic models for latency profiling, source documents, evaluation scores, and language metadata.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrievalTimings(BaseModel):
    query_encoding_ms: float = Field(default=0.0, description="Time taken to encode dense and sparse query embeddings.")
    dense_search_ms: float = Field(default=0.0, description="FAISS HNSW dense search time in ms.")
    sparse_search_ms: float = Field(default=0.0, description="SciPy sparse lexical search time in ms.")
    rrf_ms: float = Field(default=0.0, description="Reciprocal Rank Fusion and initial sorting time in ms.")
    metadata_ms: float = Field(default=0.0, description="SQLite metadata retrieval time in ms.")
    rerank_ms: Optional[float] = Field(default=None, description="Re-ranking time in ms (if enabled).")
    retrieval_total_ms: float = Field(default=0.0, description="Total time for retrieval pipeline in ms.")


class LatencyBreakdown(BaseModel):
    stt_ms: Optional[float] = Field(None, description="Audio speech-to-text transcription latency in ms (if voice input).")
    retrieval: RetrievalTimings = Field(..., description="Detailed retrieval phase latency breakdown.")
    ttft_ms: Optional[float] = Field(None, description="Time to first token in ms (for streaming responses).")
    llm_ms: float = Field(..., description="Groq LLM generation time in ms.")
    eval_ms: Optional[float] = Field(None, description="DeepEval evaluation latency in ms.")
    tts_ms: Optional[float] = Field(None, description="Audio text-to-speech synthesis latency in ms (if voice reply).")
    total_ms: float = Field(..., description="Total end-to-end processing latency in ms.")


class SourceDocument(BaseModel):
    rank: int = Field(..., description="Rank in top-k results.")
    chunk_id: int = Field(..., description="Unique chunk ID.")
    passage_id: int = Field(..., description="Passage ID.")
    url: Optional[str] = Field(None, description="Source URL.")
    title: Optional[str] = Field(None, description="Article title.")
    section: Optional[str] = Field(None, description="Section heading.")
    text: str = Field(..., description="Text content of the retrieved chunk.")
    language: Optional[str] = Field("gu", description="Language code.")
    rrf_score: float = Field(..., description="Combined RRF hybrid score.")
    dense_rank: Optional[int] = Field(None, description="Rank from dense search.")
    sparse_rank: Optional[int] = Field(None, description="Rank from sparse search.")
    dense_score: Optional[float] = Field(None, description="Dense cosine similarity score.")
    sparse_score: Optional[float] = Field(None, description="Sparse lexical dot product score.")
    rerank_score: Optional[float] = Field(None, description="Score assigned by re-ranker (if applied).")
    rerank_rank: Optional[int] = Field(None, description="Rank after re-ranking.")
    original_rank: Optional[int] = Field(None, description="Rank prior to re-ranking.")
    rerank_reason: Optional[str] = Field(None, description="Reasoning provided by listwise re-ranker.")


class EvaluationScores(BaseModel):
    overall_score: Optional[float] = Field(None, description="Overall DeepEval composite quality score (0.0 to 1.0).")
    faithfulness: Optional[float] = Field(None, description="Groundedness in retrieved context (DeepEval FaithfulnessMetric).")
    answer_relevance: Optional[float] = Field(None, description="Direct relevance to the asked question (DeepEval AnswerRelevancyMetric).")
    answer_correctness: Optional[float] = Field(None, description="Factual alignment with golden ground-truth answer (DeepEval GEval).")
    answer_similarity: Optional[float] = Field(None, description="BGE-M3 semantic similarity with ground-truth.")
    context_recall: Optional[float] = Field(None, description="Proportion of ground-truth facts retrieved (DeepEval ContextualRecallMetric).")
    hit_rate_at_k: Optional[float] = Field(None, description="Hit rate (1.0 if golden passage in top-k).")
    explanation: Optional[str] = Field(None, description="DeepEval reasoning and judge explanation.")
    reason: Optional[str] = Field(None, description="Detailed DeepEval metric evaluation reasons.")


class EvaluationResult(BaseModel):
    is_golden: bool = Field(..., description="True if query was matched against verified Golden evaluation dataset.")
    query_id: Optional[int] = Field(None, description="Matched Golden Query ID.")
    query_type: Optional[str] = Field(None, description="Golden query type (e.g. NUMERIC, ENTITY, DESCRIPTION).")
    ground_truth_answer: Optional[str] = Field(None, description="Ground truth reference answer (if golden).")
    warning: Optional[str] = Field(None, description="Notice if query is non-golden.")
    scores: Optional[EvaluationScores] = Field(None, description="Evaluation scorecard.")


class LanguageInfo(BaseModel):
    code: str = Field(..., description="Language ISO code (e.g. 'gu', 'hi').")
    name: str = Field(..., description="English language name (e.g. 'Gujarati', 'Hindi').")
    native_name: str = Field(..., description="Native script name (e.g. 'ગુજરાતી', 'हिन्दी').")
    locale: str = Field(..., description="Speech locale (e.g. 'gu-IN', 'hi-IN').")
    indexed_passages: int = Field(..., description="Total vectors indexed in FAISS.")
    golden_dataset_records: int = Field(..., description="Total golden evaluation records.")
    is_active: bool = Field(True, description="Whether language pipeline is loaded and ready.")
