"""
Precomputed feature cache.

Extracting a log-Mel spectrogram costs roughly 40 ms per utterance: decode the
FLAC, resample, window, FFT, mel filterbank, dB, normalise. Doing that inside
the training loop means paying it once per epoch per file. For the ASVspoof
2019 LA training partition that is 25,380 x 20 epochs = half a million
extractions, which dominates the run time and does no useful work after the
first pass, because the transform is deterministic.

This module runs the transform once, in parallel across cores, and stores the
result as a memory-mapped array. Training then reads batches straight off disk
at memory-map speed and the GPU or CPU spends its time on the network instead.

Storage is float16. The features are normalised into [0, 1], where float16 has
about three decimal digits of precision - far more than the spectrogram's own
dynamic range warrants - and it halves both the file size and the bytes read
per batch. Values are cast back to float32 when a batch is assembled.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import config
from model.datasets import Corpus

log = logging.getLogger(__name__)

_WORKER_PRE = None


def _init_worker():
    """One preprocessor per worker process, built once rather than per file."""
    global _WORKER_PRE
    from backend.preprocessor import AudioPreprocessor
    _WORKER_PRE = AudioPreprocessor()


def _extract(args: Tuple[int, str]) -> Tuple[int, Optional[np.ndarray]]:
    index, path = args
    try:
        spec = _WORKER_PRE.process_single(path)      # (1, n_mels, frames)
        return index, spec.astype(np.float16)
    except Exception:                                # noqa: BLE001
        return index, None


class FeatureCache:
    """A built cache on disk: features, labels, attack ids and a valid mask."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.name = meta["name"]
        self.kind = meta["kind"]
        self.shape = tuple(meta["shape"])
        self.attacks = np.asarray(meta["attacks"])
        self.labels = np.load(self.dir / "labels.npy")
        self.valid = np.load(self.dir / "valid.npy")
        self.features = np.load(self.dir / "features.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.valid.sum())

    @property
    def valid_indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid)

    def summary(self) -> dict:
        labels = self.labels[self.valid]
        return {
            "name": self.name,
            "kind": self.kind,
            "usable": int(self.valid.sum()),
            "dropped": int((~self.valid).sum()),
            "bonafide": int((labels == 0).sum()),
            "spoof": int((labels == 1).sum()),
            "attacks": sorted({a for a in self.attacks[self.valid] if a not in ("-", "")}),
        }


def build_cache(
    corpus: Corpus,
    out_dir: Path,
    workers: Optional[int] = None,
    rebuild: bool = False,
) -> FeatureCache:
    """
    Extract every utterance in `corpus` once and store it under `out_dir`.

    Files that fail to decode are recorded in a validity mask rather than
    substituted with zeros, so a corrupt file is excluded from training instead
    of teaching the network that a blank spectrogram carries its label.
    """
    out_dir = Path(out_dir)
    if (out_dir / "meta.json").exists() and not rebuild:
        cache = FeatureCache(out_dir)
        log.info("reusing cache %s (%s)", out_dir.name, json.dumps(cache.summary()))
        return cache

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(corpus)
    shape = (n, 1, config.N_MELS, config.TARGET_FRAMES)
    workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))

    log.info("building cache for %s: %d utterances, %d workers",
             corpus.name, n, workers)

    features = np.lib.format.open_memmap(
        out_dir / "features.npy", mode="w+", dtype=np.float16, shape=shape
    )
    valid = np.zeros(n, dtype=bool)

    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for index, spec in pool.map(
            _extract, enumerate(corpus.files), chunksize=64
        ):
            if spec is not None:
                features[index] = spec
                valid[index] = True
            done += 1
            if done % 2500 == 0:
                log.info("  %6d / %d  (%4.1f%%)", done, n, 100 * done / n)

    features.flush()
    del features

    np.save(out_dir / "labels.npy", np.asarray(corpus.labels, dtype=np.float32))
    np.save(out_dir / "valid.npy", valid)
    (out_dir / "meta.json").write_text(json.dumps({
        "name": corpus.name,
        "kind": corpus.kind,
        "shape": list(shape),
        "attacks": list(corpus.attack_ids),
        "dtype": "float16",
    }), encoding="utf-8")

    cache = FeatureCache(out_dir)
    log.info("cache built: %s", json.dumps(cache.summary()))
    if (~valid).sum():
        log.warning("%d files could not be decoded and are excluded",
                    int((~valid).sum()))
    return cache
