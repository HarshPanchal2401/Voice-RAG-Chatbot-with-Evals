"""
Native Multilingual BGE-M3 Query Encoder
========================================
High-performance native PyTorch + Transformers implementation of BGE-M3.
Provides dense (1024-d) and sparse lexical query embeddings without FlagEmbedding / pyarrow dependencies.
Shared across all languages for maximum memory efficiency.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from core.config import get_default_torch_threads, BGE_MODEL_NAME, DEVICE, USE_FP16


class NativeBGEM3:
    """
    High-performance native transformers implementation of BGE-M3.
    Avoids FlagEmbedding/pyarrow crashes on Python 3.14 + Windows.
    Shared across all languages (Gujarati, Hindi, etc.) for zero VRAM waste.
    """

    def __init__(
        self,
        model_name: str = BGE_MODEL_NAME,
        use_fp16: bool = USE_FP16,
        devices: str = DEVICE,
        quantize: Optional[bool] = None,
    ):
        self.device = torch.device(devices)
        self.use_fp16 = use_fp16 and ("cuda" in str(devices))
        self._on_cpu = ("cuda" not in str(devices))

        if self._on_cpu:
            torch.set_num_threads(get_default_torch_threads())

        hf_token = os.environ.get("HF_TOKEN") or None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.backbone = AutoModel.from_pretrained(model_name, token=hf_token).to(self.device)

        # Sparse projection head
        sparse_pt = hf_hub_download(model_name, "sparse_linear.pt", token=hf_token)
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
        self._query_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _forward(self, sentences: List[str], max_length: int):
        inputs = self.tokenizer(
            sentences,
            padding=len(sentences) > 1,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.backbone(**inputs, return_dict=True).last_hidden_state
        return inputs, hidden

    def encode(
        self,
        sentences,
        batch_size: int = 1,
        max_length: int = 48,
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
            return ids, weights
        order = np.argsort(ids, kind="stable")
        ids, weights = ids[order], weights[order]
        uniq, starts = np.unique(ids, return_index=True)
        return uniq, np.maximum.reduceat(weights, starts).astype(np.float32)

    def encode_hybrid(self, text: str, max_length: int = 48) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-query fast path with in-memory bounded LRU cache (1 MB max)."""
        cache_key = (text.strip(), max_length)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        inputs, hidden = self._forward([text], max_length)
        with torch.inference_mode():
            dense = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1).float().cpu().numpy()
            raw = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu().numpy()[0]
        ids = inputs["input_ids"].cpu().numpy()[0]
        mask = inputs["attention_mask"].cpu().numpy()[0].astype(bool)
        tok_ids, tok_w = self._sparse_arrays(ids, raw, mask)
        res = (np.ascontiguousarray(dense, dtype=np.float32), tok_ids, tok_w)

        # Evict oldest entry if cache exceeds 256 entries (~1 MB RAM max)
        if len(self._query_cache) >= 256:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = res
        return res

    def encode_query(self, query: str, max_length: int = 256):
        res = self.encode(query, max_length=max_length, return_dense=True, return_sparse=False)
        return res["dense_vecs"]
