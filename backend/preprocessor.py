"""
Audio front-end: waveform standardisation and log-Mel spectrogram extraction.

This module is imported by the FastAPI service, by the offline training
pipeline and by the Colab notebook. Keeping one implementation is deliberate:
if training sees a different feature than inference does, offline metrics stop
predicting online behaviour.
"""

from __future__ import annotations

import io
import base64
import logging
from typing import List, Tuple

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")                       # Headless rendering on a server.
from matplotlib.figure import Figure        # Figure API, not pyplot: pyplot
                                            # keeps global state and is not
                                            # safe to call from the worker
                                            # threads FastAPI runs handlers on.

import config

log = logging.getLogger(__name__)


class AudioPreprocessor:
    """Turns arbitrary input audio into fixed-size normalised log-Mel images."""

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        window_seconds: float = config.WINDOW_SECONDS,
        n_mels: int = config.N_MELS,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        target_frames: int = config.TARGET_FRAMES,
        fmin: int = config.FMIN,
        fmax: int = config.FMAX,
    ):
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.window_samples = int(sample_rate * window_seconds)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.fmin = fmin
        self.fmax = fmax

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_waveform(self, audio_source) -> np.ndarray:
        """
        Decode a path or file-like object to a mono float32 waveform at the
        configured sample rate. Length is preserved; windowing happens later.
        """
        try:
            y, _ = librosa.load(audio_source, sr=self.sample_rate, mono=True)
        except Exception as exc:                      # noqa: BLE001
            # The decoder's own message embeds the absolute path of the
            # temporary file, so it is logged rather than returned. The caller
            # gets a description of the problem, not our filesystem layout.
            log.info("decode failed: %s", exc)
            raise ValueError(
                "the file could not be decoded as audio; it may be corrupt or "
                "use an unsupported codec"
            ) from exc

        if y is None or y.size == 0:
            raise ValueError("decoded audio contains no samples")

        y = np.asarray(y, dtype=np.float32)

        duration = len(y) / float(self.sample_rate)
        if duration > config.MAX_DECODED_SECONDS:
            raise ValueError(
                f"audio is {duration:.0f}s; limit is {config.MAX_DECODED_SECONDS:.0f}s"
            )

        # Peak-normalise so that recording gain does not shift the spectrogram
        # dynamic range. Near-silence is left alone rather than amplified into
        # a wall of noise.
        peak = float(np.max(np.abs(y)))
        if peak > 1e-4:
            y = y / peak
        return y

    # ------------------------------------------------------------------
    # Silence
    # ------------------------------------------------------------------
    def trim_silence(self, y: np.ndarray) -> np.ndarray:
        """
        Remove leading and trailing silence.

        In ASVspoof 2019 LA, genuine clips carry about three times more
        surrounding silence than spoofed ones, and a model trained on the raw
        files learns that difference instead of the speech (see Mueller et al.,
        "Speech is Silver, Silence is Golden", 2021). Measured on our first
        model: trimming the silence from the same evaluation clips raised its
        EER from 9.7% to 32.0%. Trimming in both training and inference removes
        the shortcut. Clips that would become very short are left untouched.
        """
        trimmed, _ = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB,
                                         frame_length=512, hop_length=128)
        if len(trimmed) < int(config.MIN_SPEECH_SECONDS * self.sample_rate):
            return y
        peak = float(np.max(np.abs(trimmed)))
        return trimmed / peak if peak > 1e-4 else trimmed

    def fit_length(self, y: np.ndarray, n: int) -> np.ndarray:
        """
        Make a clip exactly n samples. Short clips are repeated rather than
        padded with zeros, so padding cannot reintroduce a silence cue.
        """
        if len(y) >= n:
            start = (len(y) - n) // 2
            return y[start:start + n]
        reps = int(np.ceil(n / max(len(y), 1)))
        return np.tile(y, reps)[:n]

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------
    def split_windows(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Slice a waveform into fixed-length overlapping analysis windows.

        A single 3 s window was the original design, which silently discarded
        everything after the third second of a recording. Splitting lets the
        detector find a spoofed segment anywhere in the clip. Windows that are
        mostly pause (under a tenth of the loudest window's energy) are dropped,
        because a pause carries no evidence either way.
        """
        n = self.window_samples
        if len(y) <= n:
            return [self.fit_length(y, n)]

        step = max(1, int(n * (1.0 - config.WINDOW_OVERLAP)))
        windows: List[np.ndarray] = []
        for start in range(0, len(y) - n + 1, step):
            windows.append(y[start:start + n])
            if len(windows) >= config.MAX_WINDOWS_PER_CLIP:
                break

        # Keep the tail if the final partial window still holds unseen signal.
        covered = (len(windows) - 1) * step + n
        if len(windows) < config.MAX_WINDOWS_PER_CLIP and covered < len(y):
            windows.append(y[-n:])

        rms = np.array([float(np.sqrt(np.mean(w ** 2))) for w in windows])
        keep = rms >= config.SILENT_WINDOW_RATIO * rms.max()
        return [w for w, k in zip(windows, keep) if k] or windows[:1]

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def mel_spectrogram(self, window: np.ndarray) -> np.ndarray:
        """One fixed-length window -> normalised (n_mels, target_frames) image."""
        mel = librosa.feature.melspectrogram(
            y=window,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad with the floor value, not zero: 0 dB is the loudest possible bin
        # and would paint a bright stripe that the CNN learns to key on.
        if mel_db.shape[1] < self.target_frames:
            pad = self.target_frames - mel_db.shape[1]
            mel_db = np.pad(
                mel_db, ((0, 0), (0, pad)), mode="constant",
                constant_values=float(mel_db.min()),
            )
        elif mel_db.shape[1] > self.target_frames:
            mel_db = mel_db[:, : self.target_frames]

        lo, hi = float(mel_db.min()), float(mel_db.max())
        if hi - lo > 1e-6:
            mel_db = (mel_db - lo) / (hi - lo)
        else:
            mel_db = np.zeros_like(mel_db)
        return mel_db.astype(np.float32)

    def process(self, audio_source) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline for inference.

        Returns
        -------
        waveform : (n_samples,) float32
        batch    : (n_windows, 1, n_mels, target_frames) float32, CNN-ready
        preview  : (n_mels, target_frames) float32, first window, for display
        """
        waveform = self.load_waveform(audio_source)
        windows = self.split_windows(self.trim_silence(waveform))
        specs = np.stack([self.mel_spectrogram(w) for w in windows])
        batch = specs[:, np.newaxis, :, :]
        return waveform, batch.astype(np.float32), specs[0]

    def process_single(self, audio_source) -> np.ndarray:
        """
        Training-time path: one clip -> one (1, n_mels, target_frames) tensor.

        Training uses the centre window so that every utterance contributes
        exactly one example and the class balance stays exactly as the
        protocol file declares it.
        """
        waveform = self.trim_silence(self.load_waveform(audio_source))
        window = self.fit_length(waveform, self.window_samples)
        return self.mel_spectrogram(window)[np.newaxis, :, :]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def spectrogram_png_base64(self, spectrogram: np.ndarray) -> str:
        """Render a spectrogram as a base64 PNG data URI for the dashboard."""
        fig = Figure(figsize=(4, 3), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0f172a")
        ax.imshow(spectrogram, aspect="auto", origin="lower",
                  cmap="inferno", interpolation="nearest")
        ax.axis("off")
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
