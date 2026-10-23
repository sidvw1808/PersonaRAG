"""Vector store and retrieval.

One Chroma collection per persona, embedded with a sentence-transformers model. Retrieval runs
in one of three modes (config.RETRIEVAL_MODE):

    vector         embedding similarity only
    hybrid         BM25 keyword search + embedding search, merged with Reciprocal Rank Fusion
    hybrid_rerank  hybrid candidates re-ordered by a cross-encoder that reads query and passage together
"""

import re
from dataclasses import dataclass
from functools import cached_property

import chromadb
import numpy as np

from . import config
from .chunking import Chunk
from .config import DB_DIR, EMBED_MODEL, EMBED_QUERY_PREFIX, MIN_SIMILARITY, TOP_K

RRF_K = 60  # standard Reciprocal Rank Fusion constant

_STOPWORDS = set(
    "a an and are as at be but by did do does for from had has have he her his how i if in into is it "
    "its me my of on or our she so that the their them they this to was we were what when where which "
    "who why will with would you your about can could should think".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity to the query, higher is better
    rerank: float | None = None  # cross-encoder relevance logit, when re-ranking ran

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


def rrf_fuse(*rankings: list[str]) -> list[str]:
    """Merge ranked id lists: each list contributes 1 / (RRF_K + rank) to an id's score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


class VectorStore:
    def __init__(self, path=DB_DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self._bm25: dict[str, tuple] = {}

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBED_MODEL)

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(config.RERANK_MODEL, max_length=config.RERANK_MAX_TOKENS)

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
        self._bm25.pop(persona_id, None)
        try:
            self.client.delete_collection(f"persona_{persona_id}")
        except Exception:  # noqa: BLE001 - collection did not exist yet
            pass

    def add(self, persona_id: str, chunks: list[Chunk], batch_size: int = 128) -> None:
        self._bm25.pop(persona_id, None)
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

    # ---------------- retrieval ----------------

    def _bm25_index(self, persona_id: str):
        """BM25 index over a persona's chunks, rebuilt only when the collection size changes."""
        from rank_bm25 import BM25Okapi

        col = self.collection(persona_id)
        n = col.count()
        cached = self._bm25.get(persona_id)
        if cached and cached[0] == n:
            return cached[1], cached[2]
        got = col.get(include=["documents"])
        index = BM25Okapi([tokenize(d) for d in got["documents"]])
        self._bm25[persona_id] = (n, got["ids"], index)
        return got["ids"], index

    def _keyword_ranking(self, persona_id: str, query: str, n: int) -> list[str]:
        terms = tokenize(query)
        if not terms:
            return []
        ids, index = self._bm25_index(persona_id)
        scores = index.get_scores(terms)
        order = np.argsort(scores)[::-1][:n]
        return [ids[i] for i in order if scores[i] > 0]

    def search(
        self,
        persona_id: str,
        query: str,
        k: int = TOP_K,
        min_score: float = MIN_SIMILARITY,
        mode: str | None = None,
    ) -> list[Hit]:
        mode = mode or config.RETRIEVAL_MODE
        col = self.collection(persona_id)
        total = col.count()
        if total == 0:
            return []
        qvec = np.array(self.embed([query], is_query=True)[0])
        pool = min(total, k if mode == "vector" else config.CANDIDATES)

        res = col.query(
            query_embeddings=[qvec.tolist()],
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        hits = {
            i: Hit(id=i, text=d, metadata=m, score=1.0 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        }
        vector_ranking = res["ids"][0]

        if mode == "vector":
            ordered = vector_ranking
        else:
            keyword_ranking = self._keyword_ranking(persona_id, query, pool)
            missing = [i for i in keyword_ranking if i not in hits]
            if missing:
                # Keyword-only matches still need a cosine score for the relevance gate.
                got = col.get(ids=missing, include=["documents", "metadatas", "embeddings"])
                for i, d, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
                    hits[i] = Hit(id=i, text=d, metadata=m, score=float(np.dot(qvec, e)))
            ordered = rrf_fuse(vector_ranking, keyword_ranking)

        # Relevance gate: drop passages that are semantically unrelated to the question,
        # whichever retriever found them. This is what lets the model abstain.
        candidates = [hits[i] for i in ordered if hits[i].score >= min_score]

        if mode == "hybrid_rerank" and len(candidates) > 1:
            head = candidates[: config.RERANK_POOL]
            scores = self.reranker.predict([(query, h.text) for h in head])
            for h, s in zip(head, scores):
                h.rerank = float(s)
            candidates = sorted(head, key=lambda h: h.rerank, reverse=True)

        return candidates[:k]
