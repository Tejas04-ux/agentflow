"""Embedding backends. All return L2-normalised float32 vectors (cosine == inner product)."""
from __future__ import annotations

import hashlib
import re

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype("float32")


class FastEmbedBackend:
    """Local ONNX embedding model (no GPU / torch needed)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts)), dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.query_embed([text])), dtype="float32"))


class HashBackend:
    """Deterministic bag-of-words hashing embeddings. Offline; used for tests and quick demos."""

    dim = 512
    name = "hash:512"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])


def get_embedder(backend: str, model: str):
    if backend == "hash":
        return HashBackend()
    if backend == "fastembed":
        return FastEmbedBackend(model)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use 'fastembed' or 'hash')")
