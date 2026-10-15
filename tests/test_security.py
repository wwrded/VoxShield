"""Tests for the input-handling defences."""

from pathlib import Path

import pytest

import config
from backend import security


@pytest.fixture()
def samples_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "test_samples"
    directory.mkdir()
    (directory / "clip_one.wav").write_bytes(b"RIFF" + b"\x00" * 64)
    (tmp_path / "secret.txt").write_text("credentials")
    return directory


class TestSampleResolution:
    def test_accepts_a_real_sample(self, samples_dir: Path):
        resolved = security.resolve_sample("clip_one.wav", samples_dir)
        assert resolved.name == "clip_one.wav"

    @pytest.mark.parametrize(
        "attack",
        [
            "../secret.txt",
            "..\\secret.txt",
            "../../Windows/System32/drivers/etc/hosts",
            "subdir/clip_one.wav",
            "..%2Fsecret.txt",
            "....//secret.txt",
            "",
            ".",
            "clip_one.wav\x00.txt",
        ],
    )
    def test_rejects_traversal_and_junk(self, samples_dir: Path, attack: str):
        """The v1 endpoint built a path straight from this value."""
        with pytest.raises(security.UploadRejected):
            security.resolve_sample(attack, samples_dir)

    def test_rejects_non_wav_inside_the_directory(self, samples_dir: Path):
        (samples_dir / "notes.md").write_text("hello")
        with pytest.raises(security.UploadRejected):
            security.resolve_sample("notes.md", samples_dir)

    def test_rejects_missing_file(self, samples_dir: Path):
        with pytest.raises(security.UploadRejected):
            security.resolve_sample("absent.wav", samples_dir)


class TestExtensionAllowlist:
    @pytest.mark.parametrize("name", ["call.wav", "CALL.WAV", "a.flac", "b.mp3"])
    def test_accepts_supported_audio(self, name: str):
        assert security.validate_extension(name) in config.ALLOWED_EXTENSIONS

    @pytest.mark.parametrize("name", ["payload.exe", "script.py", "archive.zip", "noext"])
    def test_rejects_everything_else(self, name: str):
        with pytest.raises(security.UploadRejected):
            security.validate_extension(name)

    def test_rejects_missing_filename(self):
        with pytest.raises(security.UploadRejected):
            security.validate_extension(None)


class TestPayloadValidation:
    def test_rejects_empty(self):
        with pytest.raises(security.UploadRejected, match="empty"):
            security.validate_payload(b"", ".wav")

    def test_rejects_oversize(self):
        payload = b"RIFF" + b"\x00" * (config.MAX_UPLOAD_BYTES + 16)
        with pytest.raises(security.UploadRejected, match="upload limit"):
            security.validate_payload(payload, ".wav")

    def test_rejects_extension_content_mismatch(self):
        """A Windows executable renamed to .wav must not reach the decoder."""
        with pytest.raises(security.UploadRejected, match="not a recognised"):
            security.validate_payload(b"MZ\x90\x00" + b"\x00" * 64, ".wav")

    @pytest.mark.parametrize(
        "magic",
        [b"RIFF", b"fLaC", b"OggS", b"ID3\x03", b"\x1a\x45\xdf\xa3", b"\xff\xfb"],
    )
    def test_accepts_known_containers(self, magic: bytes):
        security.validate_payload(magic + b"\x00" * 64, ".wav")

    def test_accepts_mp4_ftyp_box(self):
        security.validate_payload(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 32, ".m4a")


class TestTempSuffix:
    def test_passes_through_allowed(self):
        assert security.safe_temp_suffix(".flac") == ".flac"

    def test_falls_back_for_anything_else(self):
        assert security.safe_temp_suffix("/../../evil") == ".wav"
