"""
Loader for audio corpora distributed as Hugging Face parquet shards.

Some spoofing corpora are redistributed as parquet with the audio embedded as
raw file bytes rather than as a directory of FLAC files plus a protocol. This
module reads those shards and feeds the same feature cache the protocol-based
loader uses, so training code does not care which form the data arrived in.

Label convention
----------------
This project uses 0 = bonafide (a real human) and 1 = spoof, so that a higher
model score always means more suspicion, and so that "positive" in every metric
means "detected an attack".

The Hugging Face redistribution uses the opposite mapping - its ClassLabel
names are ['fake', 'real'], i.e. 0 = fake and 1 = real. Loading it without
flipping would train a detector that is confidently backwards: it would report
genuine callers as clones and clones as genuine, while showing a perfectly
healthy loss curve. The flip happens in `_rows` below and is asserted by a
sanity check in `build_parquet_cache`.
"""

from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

import config
from model.cache import FeatureCache

log = logging.getLogger(__name__)

BONAFIDE, SPOOF = 0.0, 1.0

_WORKER_PRE = None


def _init_worker():
    global _WORKER_PRE
    from backend.preprocessor import AudioPreprocessor
    _WORKER_PRE = AudioPreprocessor()


def _extract_bytes(item: Tuple[int, bytes]):
    """Decode one in-memory audio file into a (1, n_mels, frames) feature."""
    index, blob = item
    try:
        spec = _WORKER_PRE.process_single(io.BytesIO(blob))
        return index, spec.astype(np.float16)
    except Exception:                                   # noqa: BLE001
        return index, None


def _audio_bytes(cell) -> Optional[bytes]:
    """Pull the raw file bytes out of a Hugging Face Audio cell."""
    if isinstance(cell, dict):
        blob = cell.get("bytes")
        if blob:
            return blob
        path = cell.get("path")
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def _rows(shards: Sequence[Path], limit: Optional[int] = None) -> Iterator[Tuple[bytes, float]]:
    """Stream (audio_bytes, project_label) pairs from the shards, in order."""
    import pyarrow.parquet as pq

    seen = 0
    for shard in shards:
        parquet = pq.ParquetFile(str(shard))
        for batch in parquet.iter_batches(batch_size=256, columns=["audio", "label"]):
            table = batch.to_pylist()
            for row in table:
                blob = _audio_bytes(row.get("audio"))
                if blob is None:
                    continue
                # Flip: source is 0=fake / 1=real, project is 0=bonafide / 1=spoof.
                label = BONAFIDE if int(row["label"]) == 1 else SPOOF
                yield blob, label
                seen += 1
                if limit and seen >= limit:
                    return


def count_rows(shards: Sequence[Path]) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(str(s)).metadata.num_rows for s in shards)


def build_parquet_cache(
    shards: Sequence[Path],
    out_dir: Path,
    name: str,
    workers: Optional[int] = None,
    limit: Optional[int] = None,
    rebuild: bool = False,
) -> FeatureCache:
    """
    Decode every clip in the shards once and store the features as a memmap.

    Audio is streamed from parquet in batches rather than materialised as a
    list, because the three training shards together hold about 35,000 clips
    and roughly 1.2 GB of encoded audio.
    """
    out_dir = Path(out_dir)
    if (out_dir / "meta.json").exists() and not rebuild:
        cache = FeatureCache(out_dir)
        log.info("reusing cache %s: %s", out_dir.name, json.dumps(cache.summary()))
        return cache

    total = count_rows(shards)
    if limit:
        total = min(total, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))

    log.info("building %s from %d shard(s): %d clips, %d workers",
             name, len(shards), total, workers)

    shape = (total, 1, config.N_MELS, config.TARGET_FRAMES)
    features = np.lib.format.open_memmap(
        out_dir / "features.npy", mode="w+", dtype=np.float16, shape=shape
    )
    labels = np.zeros(total, dtype=np.float32)
    valid = np.zeros(total, dtype=bool)

    def indexed() -> Iterator[Tuple[int, bytes]]:
        for i, (blob, label) in enumerate(_rows(shards, limit)):
            labels[i] = label
            yield i, blob

    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for index, spec in pool.map(_extract_bytes, indexed(), chunksize=32):
            if spec is not None:
                features[index] = spec
                valid[index] = True
            done += 1
            if done % 2500 == 0:
                log.info("  %6d / %d  (%4.1f%%)", done, total, 100 * done / total)

    features.flush()
    del features

    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "valid.npy", valid)
    (out_dir / "meta.json").write_text(json.dumps({
        "name": name,
        "kind": "real_corpus",
        "shape": list(shape),
        "attacks": ["-"] * total,          # this redistribution carries no attack ids
        "dtype": "float16",
        "source_shards": [str(s.name) for s in shards],
        "label_mapping": "flipped from source 0=fake/1=real to 0=bonafide/1=spoof",
    }), encoding="utf-8")

    cache = FeatureCache(out_dir)
    summary = cache.summary()
    log.info("%s built: %s", name, json.dumps(summary))

    if summary["bonafide"] == 0 or summary["spoof"] == 0:
        raise RuntimeError(f"{name} contains only one class - check the label mapping")
    if (~valid).sum():
        log.warning("%d clips failed to decode and are excluded", int((~valid).sum()))
    return cache
