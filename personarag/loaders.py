"""Turn raw sources (files, web pages, YouTube transcripts) into plain-text Documents.

Fetched sources are cached under personas/<id>/cache/ so re-ingesting never re-downloads.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config

TEXT_EXTS = {".txt", ".md"}
SUBTITLE_EXTS = {".srt", ".vtt"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"}
SUPPORTED_EXTS = TEXT_EXTS | SUBTITLE_EXTS | AUDIO_EXTS | {".pdf"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)  # source_id, title, kind, date, url


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:60]


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Optional YAML header between '---' lines lets a .txt/.md file carry its own title/date/url."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                return yaml.safe_load(parts[1]) or {}, parts[2]
            except yaml.YAMLError:
                pass
    return {}, text


def _strip_subtitles(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or line.isdigit() or "-->" in line:
            continue
        lines.append(re.sub(r"<[^>]+>", "", line))
    # Subtitle files repeat lines across cues; drop consecutive duplicates.
    deduped = [l for i, l in enumerate(lines) if i == 0 or l != lines[i - 1]]
    return " ".join(deduped)


def load_file(path: Path, meta: dict | None = None) -> Document:
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        from .transcribe import transcribe_file

        text = transcribe_file(path)
        header = {"kind": "podcast", "transcript": "whisper"}
    elif ext == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        header = {}
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        header, text = _split_front_matter(raw)
        if ext in SUBTITLE_EXTS:
            text = _strip_subtitles(text)
    metadata = {
        "source_id": _slug(path.stem),
        "title": path.stem.replace("_", " ").title(),
        "kind": path.parent.name if path.parent.name != "raw" else "document",
        **header,
        **(meta or {}),
    }
    return Document(text=text, metadata=metadata)


def _cached(cache_dir: Path, key: str, fetch) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def fetch_url(url: str, cache_dir: Path) -> dict:
    def fetch():
        import trafilatura

        html = trafilatura.fetch_url(url)
        if not html:
            raise RuntimeError(f"Could not download {url}")
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if not text:
            raise RuntimeError(f"No article text found at {url}")
        meta = trafilatura.extract_metadata(html)
        return {
            "text": text,
            "title": (meta.title if meta and meta.title else url),
            "date": (meta.date if meta and meta.date else ""),
        }

    return _cached(cache_dir, url, fetch)


def fetch_youtube(video_id: str, cache_dir: Path, force_whisper: bool = False) -> dict:
    """Captions if the video has them; otherwise download the audio and transcribe it."""

    def fetch():
        if not force_whisper:
            from youtube_transcript_api import YouTubeTranscriptApi

            try:
                transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-IN", "en-US"])
                text = " ".join(s.text.replace("\n", " ") for s in transcript)
                method = "auto captions" if transcript.is_generated else "captions"
                return {"text": text, "title": f"YouTube {video_id}", "date": "", "method": method}
            except Exception as exc:  # noqa: BLE001 - fall through to speech-to-text
                if not config.WHISPER_FALLBACK:
                    raise RuntimeError(f"no captions ({type(exc).__name__}); Whisper fallback is off") from exc
                print(f"  no captions for {video_id} ({type(exc).__name__}); transcribing audio...", flush=True)
        from .transcribe import transcribe_url

        text = transcribe_url(f"https://www.youtube.com/watch?v={video_id}")
        return {"text": text, "title": f"YouTube {video_id}", "date": "", "method": "whisper"}

    return _cached(cache_dir, f"yt:{video_id}:{'whisper' if force_whisper else 'auto'}", fetch)


def fetch_audio(source: str, cache_dir: Path, base_dir: Path) -> dict:
    """Podcast episode page, direct audio URL, or a local audio file -> Whisper transcript."""

    def fetch():
        from .transcribe import transcribe_file, transcribe_url

        if source.startswith(("http://", "https://")):
            text = transcribe_url(source)
        else:
            text = transcribe_file(base_dir / source)
        return {"text": text, "title": Path(source).stem, "date": "", "method": "whisper"}

    return _cached(cache_dir, f"audio:{source}", fetch)


def load_persona_sources(persona_dir: Path) -> tuple[list[Document], list[str]]:
    """Load everything listed in sources.yaml plus every supported file under raw/.

    Returns (documents, errors). One bad source never stops the rest from loading.
    """
    docs: list[Document] = []
    errors: list[str] = []
    cache_dir = persona_dir / "cache"
    listed_files: set[Path] = set()

    manifest = persona_dir / "sources.yaml"
    entries = []
    if manifest.exists():
        entries = (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("sources") or []

    for entry in entries:
        kind = entry.get("type", "url")
        meta = {k: v for k, v in entry.items() if k not in {"type", "path", "id"}}
        try:
            if kind == "file":
                path = persona_dir / entry["path"]
                listed_files.add(path.resolve())
                docs.append(load_file(path, meta))
                continue
            if kind == "url":
                data = fetch_url(entry["url"], cache_dir)
                source_id = entry.get("source_id") or _slug(data["title"])
            elif kind == "youtube":
                meta.pop("whisper", None)
                data = fetch_youtube(entry["id"], cache_dir, force_whisper=entry.get("whisper", False))
                meta.setdefault("url", f"https://www.youtube.com/watch?v={entry['id']}")
                meta.setdefault("kind", "video")
                source_id = entry.get("source_id") or f"yt_{entry['id']}"
            elif kind == "audio":
                src = entry.get("url") or entry["path"]
                data = fetch_audio(src, cache_dir, persona_dir)
                meta.setdefault("kind", "podcast")
                source_id = entry.get("source_id") or _slug(data["title"])
            else:
                raise ValueError(f"unknown source type '{kind}'")
            docs.append(
                Document(
                    text=data["text"],
                    metadata={
                        "source_id": source_id,
                        "title": data["title"],
                        "date": data.get("date", ""),
                        "kind": "article",
                        "transcript": data.get("method", ""),
                        **meta,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            label = entry.get("url") or entry.get("id") or entry.get("path")
            errors.append(f"{kind} {label}: {exc}")

    raw_dir = persona_dir / "raw"
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*")):
            if path.suffix.lower() in SUPPORTED_EXTS and path.resolve() not in listed_files:
                try:
                    docs.append(load_file(path))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"file {path.name}: {exc}")

    return [d for d in docs if d.text.strip()], errors
