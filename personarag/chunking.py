"""Sentence-aware chunking with overlap.

Chunks end on sentence boundaries so a retrieved passage never starts or stops mid-thought,
and consecutive chunks share a few sentences so an idea split across a boundary is still
retrievable from either side.
"""

import re
from dataclasses import dataclass

from .config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS
from .loaders import Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9(])")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_END.split(text)
    # Transcripts often have no punctuation at all; fall back to fixed word windows.
    out: list[str] = []
    for s in sentences:
        words = s.split()
        if len(words) > CHUNK_WORDS:
            step = CHUNK_WORDS // 2
            out.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
        else:
            out.append(s)
    return out


def chunk_text(
    text: str, max_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                carried += len(prev.split())
                if carried > overlap_words:
                    break
                carry.insert(0, prev)
            current, count = carry, sum(len(s.split()) for s in carry)
        current.append(sentence)
        count += n
    if current:
        tail = " ".join(current)
        if not chunks or tail not in chunks[-1]:
            chunks.append(tail)
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    source_id = doc.metadata.get("source_id", "doc")
    pieces = chunk_text(doc.text)
    return [
        Chunk(
            id=f"{source_id}-{i:04d}",
            text=piece,
            metadata={**doc.metadata, "chunk_index": i, "chunk_count": len(pieces)},
        )
        for i, piece in enumerate(pieces)
    ]
