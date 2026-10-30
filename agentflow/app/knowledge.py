"""Knowledge base used by the RAG tool: chunk -> embed -> FAISS."""
from __future__ import annotations

import uuid
from pathlib import Path

from .chunking import chunk_text
from .loaders import SUPPORTED, load_document
from .vectorstore import VectorStore


class KnowledgeBase:
    def __init__(self, data_dir: Path, embedder, chunk_size: int = 800, chunk_overlap: int = 120):
        self.embedder = embedder
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.store = VectorStore(Path(data_dir) / "index", embedder.dim, embedder.name)

    def add(self, name: str, data: bytes) -> dict:
        pieces: list[tuple[int | None, str]] = []
        for page, text in load_document(name, data):
            pieces += [(page, c) for c in chunk_text(text, self.chunk_size, self.chunk_overlap)]
        if not pieces:
            raise ValueError(f"No extractable text found in '{name}'")
        doc_id = uuid.uuid4().hex[:12]
        self.store.add_document(doc_id, name, pieces, self.embedder.embed([t for _, t in pieces]))
        return {"id": doc_id, "name": name, "chunks": len(pieces)}

    def delete(self, doc_id: str) -> bool:
        return bool(self.store.delete_document(doc_id))

    def documents(self) -> list[dict]:
        return self.store.documents()

    def is_empty(self) -> bool:
        return not self.store.chunks

    def search(self, question: str, k: int) -> list[dict]:
        hits = self.store.search(self.embedder.embed_query(question), k)
        return [
            {"n": i, "title": c["doc_name"] + (f", p.{c['page']}" if c.get("page") else ""),
             "url": None, "score": round(s, 4), "text": c["text"]}
            for i, (c, s) in enumerate(hits, start=1)
        ]

    def preload(self, directory: Path) -> int:
        """Index files from a folder that are not yet in the store. Returns number added."""
        directory = Path(directory)
        if not directory.is_dir():
            return 0
        known = {d["name"] for d in self.documents()}
        added = 0
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED and f.name not in known:
                self.add(f.name, f.read_bytes())
                added += 1
        return added
