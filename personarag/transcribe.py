"""Speech-to-text for sources that have no text: podcast audio, and videos without captions.

    audio URL / podcast page / video  --yt-dlp-->  audio file  --faster-whisper-->  transcript

Both libraries are optional: `pip install faster-whisper yt-dlp`.
"""

import tempfile
from functools import lru_cache
from pathlib import Path

from . import config


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE
    )


def download_audio(url: str, out_dir: Path) -> Path:
    """Download the audio track of anything yt-dlp supports (YouTube, podcast hosts, direct MP3)."""
    import yt_dlp

    opts = {
        # Prefer small audio-only formats; no ffmpeg needed since Whisper decodes them directly.
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def transcribe_file(path: Path) -> str:
    segments, _info = _model().transcribe(str(path), vad_filter=True, beam_size=1)
    return " ".join(s.text.strip() for s in segments)


def transcribe_url(url: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        return transcribe_file(download_audio(url, Path(tmp)))
