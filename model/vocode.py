"""
Vocoder resynthesis of real recordings, used to make matched spoofed examples.

If genuine LibriSpeech clips were added to training with no spoofed clips
from the same recordings, the model could learn "sounds like LibriSpeech, so
it is real", which is the same shortcut problem in a new place. Passing each
real clip through a vocoder gives a spoofed twin with identical speaker,
words, microphone and room; the only difference left is the vocoder trace.

This follows the idea of training on vocoded copies of genuine speech
(X. Wang and J. Yamagishi, ICASSP 2023), but uses a classical Griffin-Lim
vocoder so that it runs without a GPU. Two variants are produced:

  0  mel spectrogram (80 bands) inverted with Griffin-Lim
  1  the same, with the mel spectrogram smoothed over time first, imitating
     the over-smoothed output of a text-to-speech acoustic model

These are not modern neural voice clones. They add matched negatives in a new
recording domain; the ASVspoof attacks remain the main source of real
synthesis systems.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

import config

SR = config.SAMPLE_RATE
N_FFT, HOP, N_MELS = 1024, 256, 80


def resynthesise(y: np.ndarray, variant: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, power=2.0)
    if variant == 1:
        mel = uniform_filter1d(mel, size=3, axis=1)
    out = librosa.feature.inverse.mel_to_audio(mel, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                               power=2.0, n_iter=32)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak).astype(np.float32)
