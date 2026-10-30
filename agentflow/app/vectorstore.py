"""FAISS-backed vector store with on-disk persistence."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, directory: Path, dim: int, backend_name: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.backend_name = backend_name
        self._lock = threading.RLock()
        self.chunks: list[dict] = []
        self.vectors = np.zeros((0, dim), dtype="float32")
        self._index = faiss.IndexFlatIP(dim)
        self._load()

    # ---- persistence -------------------------------------------------
    @property
    def _meta_path(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def _vec_path(self) -> Path:
        return self.dir / "vectors.npy"

    def _load(self) -> None:
        if not (self._meta_path.exists() and self._vec_path.exists()):
            return
        meta = json.loads(self._meta_path.read_text())
        if meta.get("backend") != self.backend_name:
            raise RuntimeError(
                f"Stored index was built with '{meta.get('backend')}' but the app is configured for "
                f"'{self.backend_name}'. Use the same embedding settings or clear the data volume."
            )
        self.chunks = meta["chunks"]
        self.vectors = np.load(self._vec_path)
        self._rebuild()

    def _save(self) -> None:
        self._meta_path.write_text(json.dumps({"backend": self.backend_name, "chunks": self.chunks}))
        np.save(self._vec_path, self.vectors)

    def _rebuild(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        if len(self.vectors):
            self._index.add(self.vectors)

    # ---- operations --------------------------------------------------
    def add_document(self, doc_id: str, doc_name: str, pieces: list[tuple[int | None, str]], vectors: np.ndarray):
        """`pieces` is a list of (page, chunk_text); `vectors` has one row per piece."""
        with self._lock:
            for page, text in pieces:
                self.chunks.append(
                    {"id": uuid.uuid4().hex, "doc_id": doc_id, "doc_name": doc_name, "page": page, "text": text}
                )
            vectors = vectors.astype("float32")
            self.vectors = np.vstack([self.vectors, vectors])
            self._index.add(vectors)
            self._save()

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            removed = len(self.chunks) - len(keep)
            if removed:
                self.chunks = [self.chunks[i] for i in keep]
                self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), dtype="float32")
                self._rebuild()
                self._save()
            return removed

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        with self._lock:
            if not self.chunks:
                return []
            scores, idx = self._index.search(query_vec.astype("float32"), min(k, len(self.chunks)))
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def documents(self) -> list[dict]:
        with self._lock:
            docs: dict[str, dict] = {}
            for c in self.chunks:
                d = docs.setdefault(c["doc_id"], {"id": c["doc_id"], "name": c["doc_name"], "chunks": 0})
                d["chunks"] += 1
            return list(docs.values())
