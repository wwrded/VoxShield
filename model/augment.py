"""
Channel augmentation: make clean audio sound like it came through a real line.

Applied at random to genuine and spoofed clips alike during training, so the
model cannot use recording quality as a shortcut for the label. The same
functions drive the deployment-readiness test, with fixed settings.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

import config

SR = config.SAMPLE_RATE


def telephone(y: np.ndarray) -> np.ndarray:
    """Narrowband call: 300-3400 Hz band-pass, down to 8 kHz and back."""
    sos = sps.butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sps.sosfilt(sos, y)
    return sps.resample_poly(sps.resample_poly(y, 1, 2), 2, 1)


def g711(y: np.ndarray) -> np.ndarray:
    """Narrowband call encoded with 8-bit mu-law (G.711), then decoded."""
    y = telephone(y)
    peak = float(np.max(np.abs(y))) or 1.0
    x = np.clip(y / peak, -1, 1)
    mu = 255.0
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    enc = np.round((enc + 1) / 2 * 255) / 255 * 2 - 1
    return np.sign(enc) * (np.power(1 + mu, np.abs(enc)) - 1) / mu * peak


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(y ** 2)) or 1e-8
    return y + rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), y.shape)


def lowpass(y: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Cheap microphone or lossy compression: everything above cutoff removed."""
    sos = sps.butter(8, cutoff_hz, btype="lowpass", fs=SR, output="sos")
    return sps.sosfilt(sos, y)


def apply(y: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Apply one named channel condition. Unknown kinds return the clip unchanged."""
    if kind == "telephone":
        return telephone(y)
    if kind == "g711":
        return g711(y)
    if kind == "noise":
        return add_noise(y, rng.uniform(10, 30), rng)
    if kind == "noise20":
        return add_noise(y, 20.0, rng)
    if kind == "lowpass":
        return lowpass(y, rng.uniform(3800, 7000))
    return y


# Training mix: half the clips stay clean, the rest get one channel condition.
TRAIN_CHOICES = ["clean", "telephone", "g711", "noise", "lowpass"]
TRAIN_PROBS = [0.50, 0.15, 0.15, 0.10, 0.10]


def random_kind(rng: np.random.Generator) -> str:
    return str(rng.choice(TRAIN_CHOICES, p=TRAIN_PROBS))
