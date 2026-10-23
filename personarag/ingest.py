"""Build a persona's knowledge base: load sources -> chunk -> embed -> store.

    python -m personarag.ingest elon_musk
    python -m personarag.ingest --all
"""

import argparse
import time

from . import config
from .chunking import chunk_document
from .loaders import load_persona_sources
from .personas import list_personas, load_persona
from .store import VectorStore


def ingest(persona_id: str, store: VectorStore, reset: bool = True) -> dict:
    persona = load_persona(persona_id)
    started = time.time()
    docs, errors = load_persona_sources(persona.dir)
    chunks = [c for d in docs for c in chunk_document(d)]
    if reset:
        store.reset(persona_id)
    if chunks:
        store.add(persona_id, chunks)
    return {
        "persona": persona.name,
        "documents": len(docs),
        "chunks": len(chunks),
        "words": sum(len(d.text.split()) for d in docs),
        "errors": errors,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("persona", nargs="?", help="persona folder name, e.g. elon_musk")
    parser.add_argument("--all", action="store_true", help="ingest every persona")
    parser.add_argument("--append", action="store_true", help="keep existing chunks instead of rebuilding")
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="skip videos whose captions can't be fetched instead of transcribing their audio",
    )
    args = parser.parse_args()
    if not args.persona and not args.all:
        parser.error("give a persona id or --all")
    if args.no_whisper:
        config.WHISPER_FALLBACK = False

    ids = [p.id for p in list_personas()] if args.all else [args.persona]
    store = VectorStore()
    for pid in ids:
        report = ingest(pid, store, reset=not args.append)
        print(
            f"{report['persona']}: {report['documents']} documents, {report['words']:,} words "
            f"-> {report['chunks']} chunks in {report['seconds']}s"
        )
        for err in report["errors"]:
            print(f"  ! skipped {err}")
        if not report["chunks"]:
            print(f"  (no text found; add files to personas/{pid}/raw/ or entries to sources.yaml)")


if __name__ == "__main__":
    main()
