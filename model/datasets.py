"""
Dataset construction.

Three loaders, each with an honest name:

  asvspoof_2019_la   the real benchmark, read from its official protocol files
  folder_corpus      any directory laid out as bonafide/ and spoof/
  synthetic_demo     procedurally generated tones - a smoke test, not data

The third one exists because the pipeline has to be runnable on a laptop with
no corpus present. It is named and labelled so that a checkpoint trained on it
can never be mistaken for a trained detector: `training_data_kind` is written
into the checkpoint metadata, and the serving runtime refuses to mark such a
model TRUSTWORTHY.

ASVspoof 2019 LA layout
-----------------------
    LA/
      ASVspoof2019_LA_train/flac/LA_T_1000137.flac
      ASVspoof2019_LA_dev/flac/...
      ASVspoof2019_LA_eval/flac/...
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Each protocol line is:
    SPEAKER_ID  UTTERANCE_ID  -  ATTACK_ID  LABEL
where LABEL is `bonafide` or `spoof` and ATTACK_ID is `-` for bonafide and
A01..A19 for the spoofing systems. Label convention throughout this project is
0 = bonafide, 1 = spoof, so a higher model score always means more suspicion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

BONAFIDE, SPOOF = 0.0, 1.0

_PARTITIONS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}


@dataclass
class Corpus:
    """A list of audio paths with labels and provenance."""

    name: str
    kind: str                    # "real_corpus" or "synthetic_demo"
    files: List[str]
    labels: List[float]
    attack_ids: List[str]

    def __len__(self) -> int:
        return len(self.files)

    def summary(self) -> Dict[str, object]:
        labels = np.asarray(self.labels)
        return {
            "name": self.name,
            "kind": self.kind,
            "total": int(labels.size),
            "bonafide": int((labels == BONAFIDE).sum()),
            "spoof": int((labels == SPOOF).sum()),
            "attacks": sorted({a for a in self.attack_ids if a not in ("-", "")}),
        }


# ----------------------------------------------------------------------
def asvspoof_2019_la(
    root: Path,
    partition: str,
    limit: Optional[int] = None,
    seed: int = 1337,
) -> Corpus:
    """
    Load one ASVspoof 2019 Logical Access partition.

    `limit` subsamples in a class-stratified, seeded way. v1 truncated the
    protocol file with `if len(files) >= 3000: break`, which is not a sample -
    the protocol is grouped, so the first N lines are mostly one class and a
    handful of attack types. That silently produced an unbalanced training set
    that had never seen most of the attacks it was meant to detect.
    """
    if partition not in _PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(_PARTITIONS)}")

    audio_dirname, protocol_name = _PARTITIONS[partition]
    root = Path(root)
    protocol = root / "ASVspoof2019_LA_cm_protocols" / protocol_name
    audio_dir = root / audio_dirname / "flac"

    if not protocol.is_file():
        raise FileNotFoundError(f"protocol file not found: {protocol}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio directory not found: {audio_dir}")

    files: List[str] = []
    labels: List[float] = []
    attacks: List[str] = []
    missing = 0

    with protocol.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            utt_id, attack_id, label = parts[1], parts[3], parts[4]
            path = audio_dir / f"{utt_id}.flac"
            if not path.is_file():
                missing += 1
                continue
            files.append(str(path))
            labels.append(SPOOF if label == "spoof" else BONAFIDE)
            attacks.append(attack_id)

    if missing:
        log.warning("%d utterances listed in %s were not on disk",
                    missing, protocol.name)
    if not files:
        raise RuntimeError(f"no audio found for partition {partition} under {root}")

    corpus = Corpus(
        name=f"ASVspoof2019-LA-{partition}",
        kind="real_corpus",
        files=files,
        labels=labels,
        attack_ids=attacks,
    )

    if limit is not None and limit < len(corpus):
        corpus = stratified_subsample(corpus, limit, seed)
    return corpus


def stratified_subsample(corpus: Corpus, limit: int, seed: int = 1337) -> Corpus:
    """
    Take `limit` items while preserving the class ratio and attack coverage.

    ASVspoof LA is roughly 1:9 bonafide:spoof. Preserving that ratio keeps the
    subsample representative; drawing spoof items round-robin across attack
    ids keeps every attack type present even in a small subset.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(corpus.labels)

    bona_idx = np.flatnonzero(labels == BONAFIDE)
    spoof_idx = np.flatnonzero(labels == SPOOF)

    bona_share = len(bona_idx) / len(labels)
    n_bona = max(1, min(len(bona_idx), int(round(limit * bona_share))))
    n_spoof = max(1, min(len(spoof_idx), limit - n_bona))

    chosen = list(rng.choice(bona_idx, size=n_bona, replace=False))

    # Round-robin across attack ids so no attack disappears from the subset.
    by_attack: Dict[str, List[int]] = {}
    for i in spoof_idx:
        by_attack.setdefault(corpus.attack_ids[i], []).append(int(i))
    for pool in by_attack.values():
        rng.shuffle(pool)

    keys = sorted(by_attack)
    cursor = 0
    while len(chosen) < n_bona + n_spoof and keys:
        key = keys[cursor % len(keys)]
        if by_attack[key]:
            chosen.append(by_attack[key].pop())
        else:
            keys.remove(key)
            continue
        cursor += 1

    chosen = sorted(int(i) for i in chosen)
    return Corpus(
        name=f"{corpus.name}-sub{len(chosen)}",
        kind=corpus.kind,
        files=[corpus.files[i] for i in chosen],
        labels=[corpus.labels[i] for i in chosen],
        attack_ids=[corpus.attack_ids[i] for i in chosen],
    )


# ----------------------------------------------------------------------
def folder_corpus(root: Path, name: str = "custom") -> Corpus:
    """
    Load a corpus laid out as two directories.

        root/bonafide/*.wav|flac|mp3|ogg
        root/spoof/*.wav|flac|mp3|ogg

    This is the path for locally collected data - for example the code-mixed
    Sinhala/Tamil/English recordings the proposal identifies as the research
    gap, paired against clones generated from them.
    """
    root = Path(root)
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
    files: List[str] = []
    labels: List[float] = []

    for sub, label in (("bonafide", BONAFIDE), ("spoof", SPOOF)):
        directory = root / sub
        if not directory.is_dir():
            raise FileNotFoundError(f"expected directory {directory}")
        found = sorted(p for ext in exts for p in directory.glob(ext))
        if not found:
            raise RuntimeError(f"no audio files under {directory}")
        files.extend(str(p) for p in found)
        labels.extend([label] * len(found))

    return Corpus(
        name=name,
        kind="real_corpus",
        files=files,
        labels=labels,
        attack_ids=["-"] * len(files),
    )


# ----------------------------------------------------------------------
def synthetic_demo(
    out_dir: Path,
    per_class: int = 150,
    seed: int = 1337,
    sample_rate: int = 16_000,
    seconds: float = 3.0,
) -> Corpus:
    """
    Generate two families of oscillator tones.

    THIS IS NOT SPEECH. It is a smoke test that proves the training loop, the
    data loader and the metric code run. A classifier trained on it learns to
    separate "frequency-modulated harmonic stack plus noise" from "rigid
    harmonic stack plus square-wave gating", which is a trivially separable
    problem and has no bearing on detecting a real voice clone.

    Any checkpoint produced from this corpus is stamped
    training_data_kind="synthetic_demo" and the API refuses to describe it as
    a trained detector.
    """
    import soundfile as sf

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "bonafide").mkdir(parents=True, exist_ok=True)
    (out_dir / "spoof").mkdir(parents=True, exist_ok=True)

    n = int(sample_rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)

    files: List[str] = []
    labels: List[float] = []

    for i in range(per_class):
        f0 = rng.uniform(110, 230) + rng.uniform(5, 12) * np.sin(
            2 * np.pi * rng.uniform(2, 6) * t
        )
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        audio = (
            np.sin(phase)
            + 0.50 * np.sin(2 * phase)
            + 0.25 * np.sin(3 * phase)
            + 0.12 * np.sin(4 * phase)
        )
        audio += 0.35 * np.sin(2 * np.pi * rng.uniform(450, 650) * t)
        audio += 0.20 * np.sin(2 * np.pi * rng.uniform(1400, 1800) * t)
        audio *= (np.sin(2 * np.pi * rng.uniform(1.2, 2.8) * t) ** 2) + 0.08
        audio += rng.normal(0, 0.015, n)
        audio /= np.max(np.abs(audio)) + 1e-6
        path = out_dir / "bonafide" / f"tone_natural_{i:04d}.wav"
        sf.write(path, (audio * 0.9).astype(np.float32), sample_rate)
        files.append(str(path))
        labels.append(BONAFIDE)

    for i in range(per_class):
        f0 = rng.uniform(120, 210)
        phase = 2 * np.pi * f0 * t
        audio = np.zeros(n)
        for h in range(1, 12):
            audio += (1.0 / h) * np.sin(h * phase + h * 0.5)
        audio *= (np.sin(2 * np.pi * rng.uniform(1.5, 2.5) * t) > 0).astype(float) * 0.85
        audio += 0.15 * np.sin(2 * np.pi * rng.uniform(3600, 4000) * t)
        audio /= np.max(np.abs(audio)) + 1e-6
        path = out_dir / "spoof" / f"tone_rigid_{i:04d}.wav"
        sf.write(path, (audio * 0.9).astype(np.float32), sample_rate)
        files.append(str(path))
        labels.append(SPOOF)

    log.warning(
        "generated %d synthetic tones - this is a smoke-test corpus, not speech",
        len(files),
    )
    return Corpus(
        name="synthetic-tone-demo",
        kind="synthetic_demo",
        files=files,
        labels=labels,
        attack_ids=["-"] * len(files),
    )


# ----------------------------------------------------------------------
def speaker_disjoint_split(
    corpus: Corpus, val_fraction: float = 0.2, seed: int = 1337
) -> Tuple[Corpus, Corpus]:
    """
    Split a corpus for validation without letting a speaker straddle the split.

    Utterances from the same speaker share channel and voice characteristics.
    If the same speaker appears on both sides, validation accuracy measures
    memorisation rather than generalisation. ASVspoof utterance ids carry the
    speaker prefix (LA_0079_...), which is what this keys on; for corpora
    without that structure it degrades to a plain file-level split.
    """
    rng = np.random.default_rng(seed)

    def speaker_of(path: str) -> str:
        stem = Path(path).stem
        parts = stem.split("_")
        return "_".join(parts[:2]) if len(parts) >= 3 else stem

    speakers = sorted({speaker_of(f) for f in corpus.files})
    if len(speakers) < 4:                      # Not enough structure to split on.
        idx = rng.permutation(len(corpus))
        cut = int(len(corpus) * (1 - val_fraction))
        return _subset(corpus, idx[:cut], "train"), _subset(corpus, idx[cut:], "val")

    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_fraction))
    val_speakers = set(speakers[:n_val])

    train_idx = [i for i, f in enumerate(corpus.files) if speaker_of(f) not in val_speakers]
    val_idx = [i for i, f in enumerate(corpus.files) if speaker_of(f) in val_speakers]
    return _subset(corpus, train_idx, "train"), _subset(corpus, val_idx, "val")


def _subset(corpus: Corpus, indices: Sequence[int], tag: str) -> Corpus:
    indices = [int(i) for i in indices]
    return Corpus(
        name=f"{corpus.name}-{tag}",
        kind=corpus.kind,
        files=[corpus.files[i] for i in indices],
        labels=[corpus.labels[i] for i in indices],
        attack_ids=[corpus.attack_ids[i] for i in indices],
    )
