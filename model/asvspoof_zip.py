"""
Read ASVspoof 2019 LA directly out of the distributed ZIP archive.

The archive holds 122,328 members and about 10 GB once expanded. Unpacking it
creates 122,000 small files, which on NTFS costs more wall-clock time than
downloading the archive did, and the files are then read exactly once while
building the feature cache. Reading members straight from the archive skips
that entirely: `zipfile` gives random access to any member, so one handle in
the parent process feeds a pool of decoders.

The protocol files are read from the same archive, so the labels and the audio
can never fall out of step.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from model.cache import FeatureCache

log = logging.getLogger(__name__)

BONAFIDE, SPOOF = 0.0, 1.0

ROOT = "LA"
PROTOCOLS = {
    "train": (f"{ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
              f"{ROOT}/ASVspoof2019_LA_train/flac"),
    "dev":   (f"{ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
              f"{ROOT}/ASVspoof2019_LA_dev/flac"),
    "eval":  (f"{ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
              f"{ROOT}/ASVspoof2019_LA_eval/flac"),
}


@dataclass
class ZipCorpus:
    """A partition of the archive: member paths, labels, attack and speaker ids."""

    name: str
    partition: str
    members: List[str]
    labels: List[float]
    attack_ids: List[str]
    speakers: List[str]
    kind: str = "real_corpus"

    def __len__(self) -> int:
        return len(self.members)

    def summary(self) -> dict:
        labels = np.asarray(self.labels)
        return {
            "name": self.name,
            "partition": self.partition,
            "total": int(labels.size),
            "bonafide": int((labels == BONAFIDE).sum()),
            "spoof": int((labels == SPOOF).sum()),
            "speakers": len(set(self.speakers)),
            "attacks": sorted({a for a in self.attack_ids if a not in ("-", "")}),
        }


def load_partition(
    zip_path: Path,
    partition: str,
    limit: Optional[int] = None,
    seed: int = 1337,
) -> ZipCorpus:
    """
    Parse one protocol file and resolve each utterance to a ZIP member.

    Protocol lines are:  SPEAKER  UTTERANCE  -  ATTACK  LABEL
    """
    if partition not in PROTOCOLS:
        raise ValueError(f"partition must be one of {sorted(PROTOCOLS)}")
    proto_name, audio_dir = PROTOCOLS[partition]

    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())
        if proto_name not in available:
            raise FileNotFoundError(f"{proto_name} not found in archive")
        text = zf.read(proto_name).decode("utf-8", "replace")

        members, labels, attacks, speakers = [], [], [], []
        missing = 0
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            speaker, utt, attack, label = parts[0], parts[1], parts[3], parts[4]
            member = f"{audio_dir}/{utt}.flac"
            if member not in available:
                missing += 1
                continue
            members.append(member)
            labels.append(SPOOF if label == "spoof" else BONAFIDE)
            attacks.append(attack)
            speakers.append(speaker)

    if missing:
        log.warning("%d utterances in the %s protocol are absent from the archive",
                    missing, partition)
    if not members:
        raise RuntimeError(f"no audio resolved for partition {partition}")

    corpus = ZipCorpus(
        name=f"ASVspoof2019-LA-{partition}",
        partition=partition,
        members=members,
        labels=labels,
        attack_ids=attacks,
        speakers=speakers,
    )
    if limit and limit < len(corpus):
        corpus = stratified_subset(corpus, limit, seed)
    return corpus


def stratified_subset(corpus: ZipCorpus, limit: int, seed: int = 1337) -> ZipCorpus:
    """
    Subsample while preserving the class ratio and keeping every attack type.

    The protocol file is grouped by speaker and attack, so taking the first N
    lines is not a sample - it is a biased slice missing most attack types.
    Spoof items are drawn round-robin across attack ids so that a small subset
    still contains every attack the partition defines.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(corpus.labels)
    bona = np.flatnonzero(labels == BONAFIDE)
    spoof = np.flatnonzero(labels == SPOOF)

    share = len(bona) / len(labels)
    n_bona = max(1, min(len(bona), int(round(limit * share))))
    n_spoof = max(1, min(len(spoof), limit - n_bona))

    chosen = list(rng.choice(bona, size=n_bona, replace=False))

    by_attack: Dict[str, List[int]] = {}
    for i in spoof:
        by_attack.setdefault(corpus.attack_ids[i], []).append(int(i))
    for pool in by_attack.values():
        rng.shuffle(pool)

    keys = sorted(by_attack)
    cursor = 0
    while len(chosen) < n_bona + n_spoof and keys:
        key = keys[cursor % len(keys)]
        if by_attack[key]:
            chosen.append(by_attack[key].pop())
            cursor += 1
        else:
            keys.remove(key)

    idx = sorted(int(i) for i in chosen)
    return ZipCorpus(
        name=f"{corpus.name}-sub{len(idx)}",
        partition=corpus.partition,
        members=[corpus.members[i] for i in idx],
        labels=[corpus.labels[i] for i in idx],
        attack_ids=[corpus.attack_ids[i] for i in idx],
        speakers=[corpus.speakers[i] for i in idx],
    )


# ----------------------------------------------------------------------
# Feature extraction straight from the archive
# ----------------------------------------------------------------------
_PRE = None


def _init_worker():
    global _PRE
    from backend.preprocessor import AudioPreprocessor
    _PRE = AudioPreprocessor()


def _extract(item: Tuple[int, bytes]):
    """Decode one member's bytes. Workers never touch the archive itself."""
    index, blob = item
    try:
        spec = _PRE.process_single(io.BytesIO(blob))
        return index, spec.astype(np.float16)
    except Exception:                                # noqa: BLE001
        return index, None


def _stream_members(zf: zipfile.ZipFile, members: Sequence[str]):
    """
    Yield (index, bytes) from a single archive handle in the parent process.

    Opening the archive inside each worker was the obvious design and the wrong
    one: `zipfile` builds an in-memory index of all 122,328 entries per handle,
    so twelve workers spent well over a gigabyte on twelve copies of the same
    index and then ran the machine out of memory. Reading the members here and
    shipping the bytes keeps exactly one index in existence; decompression and
    the FFT, which is where the time actually goes, still happen in parallel.
    """
    for index, member in enumerate(members):
        try:
            yield index, zf.read(member)
        except Exception:                            # noqa: BLE001
            yield index, b""


def build_zip_cache(
    zip_path: Path,
    corpus: ZipCorpus,
    out_dir: Path,
    workers: Optional[int] = None,
    rebuild: bool = False,
) -> FeatureCache:
    """Decode every member of `corpus` once and store features as a memmap."""
    out_dir = Path(out_dir)
    if (out_dir / "meta.json").exists() and not rebuild:
        cache = FeatureCache(out_dir)
        log.info("reusing cache %s: %s", out_dir.name, json.dumps(cache.summary()))
        return cache

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(corpus)
    workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))
    shape = (n, 1, config.N_MELS, config.TARGET_FRAMES)

    log.info("caching %s: %d utterances, %d workers", corpus.name, n, workers)

    features = np.lib.format.open_memmap(
        out_dir / "features.npy", mode="w+", dtype=np.float16, shape=shape)
    valid = np.zeros(n, dtype=bool)

    done = 0
    started = time.time()
    with zipfile.ZipFile(zip_path) as zf, \
            ProcessPoolExecutor(max_workers=workers,
                                initializer=_init_worker) as pool:
        # chunksize is small so the pool does not read far ahead of the
        # consumer; each queued item carries ~30 KB of audio and running too
        # far ahead is what puts pressure on memory.
        for index, spec in pool.map(_extract, _stream_members(zf, corpus.members),
                                    chunksize=8):
            if spec is not None:
                features[index] = spec
                valid[index] = True
            done += 1
            if done % 5000 == 0:
                rate = done / max(time.time() - started, 1)
                log.info("  %6d / %d  (%4.1f%%)  %.0f clips/s  eta %.1f min",
                         done, n, 100 * done / n, rate, (n - done) / rate / 60)

    features.flush()
    del features

    np.save(out_dir / "labels.npy", np.asarray(corpus.labels, dtype=np.float32))
    np.save(out_dir / "valid.npy", valid)
    (out_dir / "meta.json").write_text(json.dumps({
        "name": corpus.name,
        "kind": corpus.kind,
        "partition": corpus.partition,
        "shape": list(shape),
        "attacks": list(corpus.attack_ids),
        "speakers": list(corpus.speakers),
        "dtype": "float16",
        "source": os.path.basename(str(zip_path)),
    }), encoding="utf-8")

    cache = FeatureCache(out_dir)
    log.info("%s cached: %s", corpus.name, json.dumps(cache.summary()))
    if (~valid).sum():
        log.warning("%d utterances failed to decode and are excluded",
                    int((~valid).sum()))
    return cache
