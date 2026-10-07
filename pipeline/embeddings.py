"""
Native Multilingual BGE-M3 Query Encoder
========================================
High-performance native PyTorch + Transformers implementation of BGE-M3.
Provides dense (1024-d) and sparse lexical query embeddings without FlagEmbedding / pyarrow dependencies.
Shared across all languages for maximum memory efficiency.
"""

import os
import threading
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from core.config import (
    get_default_torch_threads,
    BGE_MODEL_NAME,
    BGE_QUERY_MAX_LENGTH,
    DEVICE,
    USE_FP16,
    QUERY_CACHE_SIZE,
)

HybridQuery = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization in float32 (same result as faiss.normalize_L2)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32)


class NativeBGEM3:
    """
    High-performance native transformers implementation of BGE-M3.
    Avoids FlagEmbedding/pyarrow crashes on Python 3.14 + Windows.
    Shared across all languages (Gujarati, Hindi, etc.) for zero VRAM waste.

    Thread-safe: tokenization is serialized (HF fast tokenizers raise "Already borrowed" when
    used concurrently) and the query cache is a lock-protected LRU that hands out copies.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fp16: bool = USE_FP16,
        devices: str = DEVICE,
        quantize: Optional[bool] = None,
        cache_size: Optional[int] = None,
    ):
        self.model_name = model_name or BGE_MODEL_NAME
        self.device = torch.device(devices)
        self.use_fp16 = use_fp16 and ("cuda" in str(devices))
        self._on_cpu = ("cuda" not in str(devices))

        if self._on_cpu:
            torch.set_num_threads(get_default_torch_threads())

        hf_token = os.environ.get("HF_TOKEN") or None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        self.backbone = AutoModel.from_pretrained(self.model_name, token=hf_token).to(self.device)

        # Sparse projection head
        sparse_pt = hf_hub_download(self.model_name, "sparse_linear.pt", token=hf_token)
        state_dict = torch.load(sparse_pt, map_location=self.device, weights_only=True)
        self.sparse_linear = nn.Linear(self.backbone.config.hidden_size, 1).to(self.device)
        self.sparse_linear.load_state_dict(state_dict)

        if self.use_fp16:
            self.backbone = self.backbone.half()
            self.sparse_linear = self.sparse_linear.half()

        self.backbone.eval()
        self.sparse_linear.eval()

        if quantize is None:
            quantize = os.environ.get("RAG_QUANTIZE_ENCODER", "0").strip().lower() in ("1", "true", "yes")
        self.quantized = bool(quantize and self._on_cpu)
        if self.quantized:
            # int8 dynamic quantization for CPU: ~2x faster, ~1GB less RAM
            self.backbone = torch.ao.quantization.quantize_dynamic(self.backbone, {nn.Linear}, dtype=torch.qint8)

        _special = [
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]
        self._special_ids = np.array([t for t in _special if t is not None], dtype=np.int64)
        self._init_cache(QUERY_CACHE_SIZE if cache_size is None else cache_size)

    # ------------------------------------------------------------------
    # Cache (lock-protected LRU; values are read-only, callers get copies)
    # ------------------------------------------------------------------

    def _init_cache(self, cache_size: int) -> None:
        self._cache_size = max(0, int(cache_size))
        self._query_cache: "OrderedDict[Tuple[str, int], HybridQuery]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._tok_lock = threading.Lock()

    @staticmethod
    def _copy_result(res: HybridQuery) -> HybridQuery:
        dense, ids, weights = res
        return dense.copy(), ids.copy(), weights.copy()

    def _cache_get(self, key: Tuple[str, int]) -> Optional[HybridQuery]:
        with self._cache_lock:
            res = self._query_cache.get(key)
            if res is None:
                return None
            self._query_cache.move_to_end(key)
        return self._copy_result(res)

    def _cache_put(self, key: Tuple[str, int], res: HybridQuery) -> None:
        if self._cache_size <= 0:
            return
        frozen = tuple(np.array(a, copy=True) for a in res)
        for a in frozen:
            a.setflags(write=False)
        with self._cache_lock:
            self._query_cache[key] = frozen  # type: ignore[assignment]
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._query_cache.clear()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _forward(self, sentences: List[str], max_length: int):
        with self._tok_lock:
            inputs = self.tokenizer(
                sentences,
                padding=len(sentences) > 1,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            hidden = self.backbone(**inputs, return_dict=True).last_hidden_state
        return inputs, hidden

    def encode(
        self,
        sentences,
        batch_size: int = 1,
        max_length: int = BGE_QUERY_MAX_LENGTH,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert_vecs: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs, hidden = self._forward(sentences, max_length)
        result: Dict[str, Any] = {}

        if return_dense:
            normed = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
            result["dense_vecs"] = normed.cpu().float().numpy()

        if return_sparse:
            with torch.inference_mode():
                raw = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu().numpy()
            input_ids = inputs["input_ids"].cpu().numpy()
            attn = inputs["attention_mask"].cpu().numpy().astype(bool)

            batch_lexical = []
            for b in range(len(sentences)):
                ids, weights = self._sparse_arrays(input_ids[b], raw[b], attn[b])
                batch_lexical.append({str(int(t)): float(w) for t, w in zip(ids, weights)})
            result["lexical_weights"] = batch_lexical

        return result

    def _sparse_arrays(self, ids: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized max-pooling of token weights per unique token id."""
        keep = mask & (weights > 0.0) & ~np.isin(ids, self._special_ids)
        ids, weights = ids[keep], weights[keep]
        if ids.size == 0:
            return ids.astype(np.int64), weights.astype(np.float32)
        order = np.argsort(ids, kind="stable")
        ids, weights = ids[order], weights[order]
        uniq, starts = np.unique(ids, return_index=True)
        return uniq.astype(np.int64), np.maximum.reduceat(weights, starts).astype(np.float32)

    def _compute_hybrid(self, text: str, max_length: int) -> HybridQuery:
        inputs, hidden = self._forward([text], max_length)
        with torch.inference_mode():
            dense = hidden[:, 0].float().cpu().numpy()
            raw = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu().numpy()[0]
        ids = inputs["input_ids"].cpu().numpy()[0]
        mask = inputs["attention_mask"].cpu().numpy()[0].astype(bool)
        tok_ids, tok_w = self._sparse_arrays(ids, raw, mask)
        return _l2_normalize_rows(dense), tok_ids, tok_w

    def encode_hybrid(self, text: str, max_length: int = BGE_QUERY_MAX_LENGTH) -> HybridQuery:
        """
        Single-query fast path: (dense [1, dim] float32 L2-normalized, sparse token ids, sparse weights).
        Results come from a bounded, thread-safe LRU cache; every call returns fresh copies, so
        callers may modify them in place (e.g. faiss.normalize_L2) without corrupting the cache.
        """
        cache_key = (text.strip(), int(max_length))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        res = self._compute_hybrid(text, max_length)
        self._cache_put(cache_key, res)
        return self._copy_result(res)

    def encode_query(self, query: str, max_length: int = 256):
        res = self.encode(query, max_length=max_length, return_dense=True, return_sparse=False)
        return res["dense_vecs"]
