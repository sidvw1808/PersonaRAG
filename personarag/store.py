"""Vector store: one Chroma collection per persona, embedded with a sentence-transformers model."""

from dataclasses import dataclass
from functools import cached_property

import chromadb

from .chunking import Chunk
from .config import DB_DIR, EMBED_MODEL, EMBED_QUERY_PREFIX, MIN_SIMILARITY, TOP_K


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity, higher is better

    @property
    def label(self) -> str:
        m = self.metadata
        parts = [m.get("title") or m.get("source_id", "source")]
        if m.get("kind"):
            parts.append(str(m["kind"]))
        if m.get("date"):
            parts.append(str(m["date"]))
        return " · ".join(parts)


def _clean_metadata(meta: dict) -> dict:
    # Chroma only stores str/int/float/bool values.
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


class VectorStore:
    def __init__(self, path=DB_DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBED_MODEL)

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query:
            texts = [EMBED_QUERY_PREFIX + t for t in texts]
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def collection(self, persona_id: str):
        return self.client.get_or_create_collection(
            f"persona_{persona_id}", metadata={"hnsw:space": "cosine"}
        )

    def reset(self, persona_id: str) -> None:
        try:
            self.client.delete_collection(f"persona_{persona_id}")
        except Exception:  # noqa: BLE001 - collection did not exist yet
            pass

    def add(self, persona_id: str, chunks: list[Chunk], batch_size: int = 128) -> None:
        col = self.collection(persona_id)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[_clean_metadata(c.metadata) for c in batch],
                embeddings=self.embed([c.text for c in batch]),
            )

    def count(self, persona_id: str) -> int:
        return self.collection(persona_id).count()

    def sources(self, persona_id: str) -> list[dict]:
        """One row per ingested source, for the UI's knowledge-base panel."""
        got = self.collection(persona_id).get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in got["metadatas"] or []:
            sid = meta.get("source_id", "?")
            row = seen.setdefault(sid, {**meta, "chunks": 0})
            row["chunks"] += 1
        return sorted(seen.values(), key=lambda r: str(r.get("title", "")))

    def search(
        self, persona_id: str, query: str, k: int = TOP_K, min_score: float = MIN_SIMILARITY
    ) -> list[Hit]:
        col = self.collection(persona_id)
        if col.count() == 0:
            return []
        res = col.query(
            query_embeddings=self.embed([query], is_query=True),
            n_results=min(k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            Hit(id=i, text=d, metadata=m, score=1.0 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        return [h for h in hits if h.score >= min_score]
