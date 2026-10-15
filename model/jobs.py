"""
Build feature caches from job descriptions.

A job names an audio source (a member of LA.zip or a file on disk) plus the
transformations to apply: optional vocoder resynthesis and optional channel
augmentation. Jobs run in parallel worker processes and the resulting
spectrograms are written to a memory-mapped cache readable by FeatureCache.

Job tuple: (index, src_kind, src_ref, vocode_variant|None, channel|None, seed, legacy)
  legacy=True reproduces the first model's front end (no silence trimming,
  zero padding) so the old checkpoint can be scored fairly on new data.
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import config

log = logging.getLogger(__name__)

LA_ZIP = Path("D:/Before_Reset/Documents/SLIIT/Y3S2/ISP_Assignment/datasets/LA.zip")
CROP_SECONDS = 4.0

_PRE = None
_ZIP = None


def _init():
    global _PRE, _ZIP
    from backend.preprocessor import AudioPreprocessor
    _PRE = AudioPreprocessor()
    _ZIP = zipfile.ZipFile(LA_ZIP) if LA_ZIP.exists() else None


def _feature(job) -> Tuple[int, Optional[np.ndarray]]:
    from model import augment, vocode
    index, kind, ref, voc, channel, seed, legacy = job
    try:
        src = io.BytesIO(_ZIP.read(ref)) if kind == "zip" else ref
        y = _PRE.load_waveform(src)
        n = _PRE.window_samples
        if legacy:
            w = np.pad(y, (0, n - len(y))) if len(y) <= n else y[(len(y) - n) // 2:(len(y) - n) // 2 + n]
            return index, _PRE.mel_spectrogram(w).astype(np.float16)
        y = _PRE.trim_silence(y)
        crop = int(CROP_SECONDS * config.SAMPLE_RATE)
        if len(y) > crop:
            s = (len(y) - crop) // 2
            y = y[s:s + crop]
        if voc is not None:
            y = vocode.resynthesise(y, int(voc))
        if channel:
            y = augment.apply(y, channel, np.random.default_rng(seed))
        y = np.asarray(y, dtype=np.float32)
        peak = float(np.max(np.abs(y)))
        if peak > 1e-4:
            y = y / peak
        return index, _PRE.mel_spectrogram(_PRE.fit_length(y, n)).astype(np.float16)
    except Exception:  # noqa: BLE001
        return index, None


def build_job_cache(name: str, jobs: Sequence[tuple], labels: Sequence[float],
                    tags: Sequence[str], out_dir: Path, workers: int = 12,
                    rebuild: bool = False):
    """Run jobs in parallel and store the features. Returns a FeatureCache."""
    from model.cache import FeatureCache
    out_dir = Path(out_dir)
    if (out_dir / "meta.json").exists() and not rebuild:
        log.info("reusing cache %s", out_dir.name)
        return FeatureCache(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(jobs)
    feats = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+", dtype=np.float16,
                                      shape=(n, 1, config.N_MELS, config.TARGET_FRAMES))
    valid = np.zeros(n, dtype=bool)
    log.info("building %s: %d items, %d workers", name, n, workers)
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for i, spec in pool.map(_feature, jobs, chunksize=32):
            if spec is not None:
                feats[i, 0] = spec
                valid[i] = True
            done += 1
            if done % 5000 == 0:
                log.info("  %s %d/%d", name, done, n)
    feats.flush()
    del feats
    np.save(out_dir / "labels.npy", np.asarray(labels, dtype=np.float32))
    np.save(out_dir / "valid.npy", valid)
    (out_dir / "meta.json").write_text(json.dumps({
        "name": name, "kind": "real_corpus",
        "shape": [n, 1, config.N_MELS, config.TARGET_FRAMES],
        "attacks": list(tags), "dtype": "float16"}), encoding="utf-8")
    if (~valid).sum():
        log.warning("%s: %d items failed", name, int((~valid).sum()))
    return FeatureCache(out_dir)
