"""Recursive text splitter with overlap."""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Break text into pieces no longer than `size`, preferring natural boundaries."""
    if len(text) <= size:
        return [text]
    if not seps:  # hard split as last resort
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep)
    out: list[str] = []
    for i, piece in enumerate(pieces):
        piece = piece + (sep if i < len(pieces) - 1 else "")
        if len(piece) > size:
            out.extend(_split(piece, size, rest))
        else:
            out.append(piece)
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into ~`size`-character chunks, each sharing `overlap` characters with the previous."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    pieces = _split(text, size - overlap, SEPARATORS)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the next chunk on a word boundary
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]
            current = tail + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
