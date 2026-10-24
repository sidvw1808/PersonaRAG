"""Portable snapshots of the knowledge bases (backup, or moving them to another machine).

    python -m personarag.snapshot export     # data/snapshot/<persona>.jsonl  (text + metadata, no vectors)
    python -m personarag.snapshot restore    # rebuild the vector store from those files

A snapshot is a backup of the knowledge bases that doesn't depend on the ChromaDB version or
its on-disk index files, and restoring it never re-downloads sources (YouTube may block
transcript requests). It holds full third-party transcripts, so data/ stays out of git.
"""

import argparse
import json

from .chunking import Chunk
from .config import ROOT
from .personas import list_personas
from .store import VectorStore

SNAPSHOT_DIR = ROOT / "data" / "snapshot"


def export(store: VectorStore) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for p in list_personas():
        got = store.collection(p.id).get(include=["documents", "metadatas"])
        rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda r: r[0])
        path = SNAPSHOT_DIR / f"{p.id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for chunk_id, text, meta in rows:
                f.write(json.dumps({"id": chunk_id, "text": text, "metadata": meta}, ensure_ascii=False) + "\n")
        print(f"{p.name}: {len(rows)} chunks -> {path.relative_to(ROOT)}")


def restore(store: VectorStore) -> None:
    for path in sorted(SNAPSHOT_DIR.glob("*.jsonl")):
        persona_id = path.stem
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        store.reset(persona_id)
        store.add(persona_id, [Chunk(r["id"], r["text"], r["metadata"]) for r in rows])
        print(f"{persona_id}: restored {len(rows)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "restore"])
    args = parser.parse_args()
    store = VectorStore()
    export(store) if args.action == "export" else restore(store)


if __name__ == "__main__":
    main()
