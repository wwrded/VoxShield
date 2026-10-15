"""
Input-handling defences for the upload endpoint.

A detection service is itself an attack surface: it accepts attacker-controlled
binary data and hands it to a media decoder written in C. The controls here are
the ones that matter for that threat model.

  * Size limit, enforced while streaming, so a large upload cannot exhaust
    memory before it is rejected.
  * Extension allowlist, so the decoder is only asked for formats we intend to
    support.
  * Magic-byte sniffing, so a .wav that is actually something else is refused
    before it reaches the decoder.
  * A sanitised temporary filename, so nothing attacker-controlled reaches a
    filesystem path.
  * Sample-id validation, so the demo-sample endpoint cannot be walked out of
    its directory.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import config


class UploadRejected(ValueError):
    """Raised when input fails validation. Message is safe to return."""


# First bytes of the container formats we accept. Ogg covers .ogg/.opus,
# "ftyp" covers MP4/M4A, 0x1A45DFA3 is Matroska/WebM. MP3 may start with an
# ID3 tag or directly with a frame sync (0xFF 0xEx/0xFx).
_MAGIC = (
    b"RIFF",       # WAV
    b"fLaC",       # FLAC
    b"OggS",       # OGG / Opus
    b"ID3",        # MP3 with ID3v2
    b"\x1a\x45\xdf\xa3",  # WebM / Matroska
)


def _looks_like_audio(head: bytes) -> bool:
    if any(head.startswith(sig) for sig in _MAGIC):
        return True
    if len(head) >= 8 and head[4:8] == b"ftyp":        # MP4 / M4A
        return True
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return True                                     # bare MP3 frame sync
    return False


def validate_extension(filename: Optional[str]) -> str:
    """Return the normalised extension, or raise UploadRejected."""
    if not filename:
        raise UploadRejected("a filename is required")

    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise UploadRejected(f"unsupported file type; accepted types are {allowed}")
    return suffix


def validate_payload(content: bytes, suffix: str) -> None:
    """Check size and container signature on the received bytes."""
    if not content:
        raise UploadRejected("the uploaded file is empty")

    if len(content) > config.MAX_UPLOAD_BYTES:
        limit_mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise UploadRejected(f"file exceeds the {limit_mb:.0f} MB upload limit")

    if not _looks_like_audio(content[:16]):
        raise UploadRejected(
            "file contents are not a recognised audio container; the extension "
            "does not match the data"
        )


_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def safe_temp_suffix(suffix: str) -> str:
    """
    Produce a suffix safe to hand to tempfile.

    v1 took everything after the last dot in the user-supplied filename and
    passed it straight into a filesystem path. This restricts the value to the
    already-validated allowlist.
    """
    return suffix if suffix in config.ALLOWED_EXTENSIONS else ".wav"


def resolve_sample(sample_id: str, samples_dir: Path) -> Path:
    """
    Resolve a demo-sample identifier to a real path inside samples_dir.

    v1 built `SAMPLES_DIR / sample_id` from a user-controlled value, which is a
    path-traversal primitive: a percent-encoded `..%2f..%2f` segment escapes
    the directory and turns the endpoint into an arbitrary-file reader. Two
    controls close it - the identifier must be a bare filename matching a
    strict pattern, and the resolved path must still sit under samples_dir
    after symlinks are followed.
    """
    normalised = unicodedata.normalize("NFKC", sample_id or "")
    if not normalised or "/" in normalised or "\\" in normalised or ".." in normalised:
        raise UploadRejected("invalid sample identifier")

    if _SAFE_STEM.search(normalised) or not normalised.lower().endswith(".wav"):
        raise UploadRejected("invalid sample identifier")

    base = samples_dir.resolve()
    candidate = (base / normalised).resolve()
    if base not in candidate.parents:
        raise UploadRejected("invalid sample identifier")
    if not candidate.is_file():
        raise UploadRejected("sample not found")
    return candidate
