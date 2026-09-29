"""
Multilingual RAG Re-ranker Module
=================================
Provides high-precision listwise re-ranking for retrieved candidate passages.
Leverages Groq's high-speed Indic/multilingual LLM (Qwen 2.5 27B / GPT-OSS) with deterministic
temperature 0.0, and includes a zero-dependency lexical cross-scoring fallback.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from groq import Groq
from dotenv import load_dotenv

from core.config import RERANKER_MODEL_NAME

load_dotenv()


class RAGReranker:
    """
    Multilingual Re-ranker for Voice & Text RAG.
    Re-scores candidate passages from initial hybrid retrieval (Dense + Sparse RRF)
    to place the most accurate, direct-answer passage at Rank 1.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        api_key: Optional[str] = None,
        timeout: float = 15.0
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip("\"' ")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def _fallback_cross_score(self, query: str, documents: List[Dict[str, Any]]) -> List[int]:
        """Deterministic lexical cross-scoring fallback if LLM call is unavailable."""
        q_words = set(re.findall(r"\w+", query.lower()))
        if not q_words:
            return list(range(len(documents)))

        scores = []
        for doc in documents:
            text = doc.get("text", "").lower()
            overlap = sum(1 for w in q_words if w in text)
            rrf = doc.get("rrf_score", 0.0)
            scores.append(overlap * 10.0 + rrf)

        ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return ranked_indices

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Re-ranks candidate documents for a given query.
        Returns: (reranked_documents, rerank_ms)
        """
        if not documents or len(documents) <= 1:
            return documents[:top_k], 0.0

        t0 = time.perf_counter()
        ranked_indices = []
        reason = ""

        if self.client:
            prompt = f"""You are an expert multilingual search re-ranker. Given a user question and candidate passages, re-rank all passages in strictly descending order of how directly and accurately they answer the question.

Question: {query}

Passages:
"""
            for i, doc in enumerate(documents):
                txt = doc.get("text", "")[:350]
                prompt += f"[Passage {i}]: {txt}\n\n"

            prompt += """Instructions:
1. Identify which passage best and most directly answers the question. That passage must be first.
2. Rank the remaining passages by decreasing relevance.
3. Return a JSON object with:
   - "ranked_indices": a list of integer indices in order of relevance (e.g. [4, 1, 0, 2, 3]).
   - "top_passage_reason": brief reason why the #1 passage was selected.
Output ONLY valid JSON."""

            candidate_models = [self.model_name, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound-mini"]
            candidate_models = list(dict.fromkeys(candidate_models))

            llm_success = False
            last_err = ""

            for m in candidate_models:
                try:
                    resp = self.client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        timeout=self.timeout
                    )
                    raw_content = resp.choices[0].message.content or "{}"
                    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
                    if fence_match:
                        raw_content = fence_match.group(1).strip()
                    elif "{" in raw_content and "}" in raw_content:
                        s_idx = raw_content.find("{")
                        e_idx = raw_content.rfind("}") + 1
                        raw_content = raw_content[s_idx:e_idx].strip()

                    parsed = json.loads(raw_content)
                    raw_indices = parsed.get("ranked_indices", [])
                    reason = parsed.get("top_passage_reason", "")

                    valid_indices = [idx for idx in raw_indices if isinstance(idx, int) and 0 <= idx < len(documents)]
                    for idx in range(len(documents)):
                        if idx not in valid_indices:
                            valid_indices.append(idx)
                    ranked_indices = valid_indices
                    llm_success = True
                    break
                except Exception as e:
                    last_err = str(e)
                    continue

            if not llm_success:
                ranked_indices = self._fallback_cross_score(query, documents)
                reason = f"Fallback cross-scoring applied ({last_err})"
        else:
            ranked_indices = self._fallback_cross_score(query, documents)
            reason = "Offline deterministic cross-scoring applied."

        rerank_ms = (time.perf_counter() - t0) * 1000

        # Construct re-ordered document list with strict ranking attributes
        reranked_docs = []
        for new_rank, idx in enumerate(ranked_indices[:top_k], start=1):
            doc_copy = dict(documents[idx])
            doc_copy["original_rank"] = doc_copy.get("rank", idx + 1)
            doc_copy["rank"] = new_rank
            doc_copy["rerank_rank"] = new_rank
            doc_copy["rerank_score"] = round(1.0 / new_rank, 4)
            if new_rank == 1 and reason:
                doc_copy["rerank_reason"] = reason
            reranked_docs.append(doc_copy)

        return reranked_docs, rerank_ms
