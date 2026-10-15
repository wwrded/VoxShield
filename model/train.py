"""
Training entry point.

Run it:

    # Real benchmark, the configuration the report should quote
    python -m model.train --dataset asvspoof --data-root ./LA --epochs 20

    # Locally collected corpus laid out as bonafide/ and spoof/
    python -m model.train --dataset folder --data-root ./my_corpus

    # Pipeline smoke test with generated tones - NOT a detector
    python -m model.train --dataset synthetic-demo --epochs 3

Every run writes two artefacts next to each other:

    model/saved_models/voice_deepfake_cnn.pth   weights
    model/saved_models/model_meta.json          what it was trained on, the
                                                calibrated threshold, and the
                                                metrics actually measured

The metadata file is what the serving runtime reads to decide whether it is
allowed to describe itself as a trained detector. Deleting it does not promote
a demo checkpoint; the runtime treats an unlabelled checkpoint as untrusted.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from backend.preprocessor import AudioPreprocessor
from model import datasets
from model.cnn_architecture import VoiceDeepfakeCNN, count_parameters
from model.evaluate import evaluate_scores, format_report, per_attack_eer, save_roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("train")


class SpectrogramDataset(Dataset):
    """
    Wraps a Corpus and produces (1, 128, 128) tensors.

    Decode failures are recorded and the item is dropped from the epoch's
    statistics rather than being replaced with a zero tensor carrying its
    original label. v1 substituted zeros on any exception, which teaches the
    network that a blank spectrogram means whatever label the unreadable file
    happened to have - a silent label-noise injection that is invisible in the
    loss curve.
    """

    def __init__(self, corpus: datasets.Corpus, preprocessor: AudioPreprocessor):
        self.corpus = corpus
        self.pre = preprocessor
        self.failures: List[str] = []

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, index: int):
        path = self.corpus.files[index]
        try:
            spec = self.pre.process_single(path)
            ok = 1.0
        except Exception as exc:                        # noqa: BLE001
            self.failures.append(f"{path}: {exc}")
            spec = np.zeros((1, config.N_MELS, config.TARGET_FRAMES), dtype=np.float32)
            ok = 0.0
        return (
            torch.from_numpy(spec),
            torch.tensor([self.corpus.labels[index]], dtype=torch.float32),
            torch.tensor(ok, dtype=torch.float32),
        )


class CachedDataset(Dataset):
    """
    Reads precomputed features from a memory-mapped cache.

    Only indices whose feature extraction succeeded are exposed, so the loop
    never sees a placeholder. Features are stored as float16 and cast to
    float32 per batch, which halves the bytes read from disk without changing
    what the network sees.
    """

    def __init__(self, cache):
        self.cache = cache
        self.index = cache.valid_indices
        self.labels = cache.labels

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        j = int(self.index[i])
        x = torch.from_numpy(np.asarray(self.cache.features[j], dtype=np.float32))
        y = torch.tensor([self.labels[j]], dtype=torch.float32)
        return x, y, torch.tensor(1.0, dtype=torch.float32)

    @property
    def class_counts(self):
        labels = self.labels[self.index]
        return int((labels == 0).sum()), int((labels == 1).sum())


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_weight = 0.0, 0.0
    scores: List[float] = []
    truths: List[float] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, targets, valid in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True).unsqueeze(1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            # Mask out items that failed to decode so they contribute no
            # gradient and no metric.
            per_item = criterion(logits, targets) * valid
            weight = valid.sum()
            if weight.item() == 0:
                continue
            loss = per_item.sum() / weight

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item()) * float(weight.item())
            total_weight += float(weight.item())

            keep = valid.squeeze(1).bool().cpu().numpy()
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            labels = targets.detach().cpu().numpy().ravel()
            scores.extend(probs[keep].tolist())
            truths.extend(labels[keep].tolist())

    mean_loss = total_loss / max(total_weight, 1.0)
    return mean_loss, np.asarray(truths), np.asarray(scores)


def train_model(
    train_corpus: datasets.Corpus,
    val_corpus: datasets.Corpus,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    patience: int = 5,
    out_dir: Optional[Path] = None,
) -> Tuple[VoiceDeepfakeCNN, Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    log.info("train: %s", json.dumps(train_corpus.summary()))
    log.info("val:   %s", json.dumps(val_corpus.summary()))

    pre = AudioPreprocessor()
    train_ds = SpectrogramDataset(train_corpus, pre)
    val_ds = SpectrogramDataset(val_corpus, pre)

    # drop_last on the training loader: the classifier head contains a
    # BatchNorm1d, which cannot compute batch statistics from a single sample
    # and raises at runtime if the final batch has size 1.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE).to(device)
    log.info("trainable parameters: %s", f"{count_parameters(model):,}")

    # ASVspoof LA is roughly 1:9 bonafide:spoof. Without a positive weight the
    # cheapest way to cut the loss is to predict "spoof" everywhere, which
    # produces a high accuracy and a useless detector. pos_weight rebalances
    # the gradient so bonafide errors cost proportionally more.
    labels = np.asarray(train_corpus.labels)
    n_spoof = max(1, int((labels == datasets.SPOOF).sum()))
    n_bona = max(1, int((labels == datasets.BONAFIDE).sum()))
    pos_weight = torch.tensor([n_bona / n_spoof], device=device)
    log.info("class balance bonafide:spoof = %d:%d, pos_weight = %.3f",
             n_bona, n_spoof, float(pos_weight))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    out_dir = Path(out_dir or config.WEIGHTS_PATH.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "voice_deepfake_cnn.pth"

    best_eer = float("inf")
    best_metrics: Dict = {}
    stale = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, y_tr, s_tr = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, y_va, s_va = _run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_loss)

        try:
            val_metrics = evaluate_scores(y_va, s_va)
        except ValueError:
            log.warning("epoch %d: validation split has a single class, skipping metrics", epoch)
            continue

        elapsed = time.perf_counter() - started
        log.info(
            "epoch %2d/%d  train_loss %.4f  val_loss %.4f  val_EER %.3f%%  "
            "val_AUC %.4f  (%.1fs)",
            epoch, epochs, train_loss, val_loss,
            val_metrics["eer_percent"], val_metrics["roc_auc"], elapsed,
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_eer_percent": val_metrics["eer_percent"],
            "val_roc_auc": round(val_metrics["roc_auc"], 5),
        })

        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), weights_path)
            log.info("  saved new best checkpoint (EER %.3f%%)", val_metrics["eer_percent"])
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("no improvement for %d epochs, stopping early", patience)
                break

    for failures, tag in ((train_ds.failures, "train"), (val_ds.failures, "val")):
        if failures:
            log.warning("%d %s files failed to decode; first: %s",
                        len(failures), tag, failures[0])

    if not best_metrics:
        raise RuntimeError("training produced no valid validation metrics")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model, {"best": best_metrics, "history": history, "weights_path": str(weights_path)}


def train_from_caches(
    train_cache,
    val_cache,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 5,
    out_dir: Optional[Path] = None,
    num_workers: int = 0,
) -> Tuple[VoiceDeepfakeCNN, Dict]:
    """Train from precomputed feature caches rather than raw audio."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CachedDataset(train_cache)
    val_ds = CachedDataset(val_cache)

    log.info("device: %s", device)
    log.info("train: %s", json.dumps(train_cache.summary()))
    log.info("val:   %s", json.dumps(val_cache.summary()))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE).to(device)
    log.info("trainable parameters: %s", f"{count_parameters(model):,}")

    n_bona, n_spoof = train_ds.class_counts
    pos_weight = torch.tensor([max(n_bona, 1) / max(n_spoof, 1)], device=device)
    log.info("class balance bonafide:spoof = %d:%d, pos_weight = %.4f",
             n_bona, n_spoof, float(pos_weight))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    out_dir = Path(out_dir or config.WEIGHTS_PATH.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "voice_deepfake_cnn.pth"

    best_eer, best_metrics, stale, history = float("inf"), {}, 0, []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, y_tr, s_tr = _run_epoch(model, train_loader, criterion,
                                            optimizer, device, True)
        val_loss, y_va, s_va = _run_epoch(model, val_loader, criterion,
                                          optimizer, device, False)
        scheduler.step(val_loss)

        try:
            val_metrics = evaluate_scores(y_va, s_va)
        except ValueError:
            log.warning("epoch %d: validation split has a single class", epoch)
            continue

        log.info("epoch %2d/%d  train_loss %.4f  val_loss %.4f  "
                 "val_EER %.3f%%  val_AUC %.4f  (%.0fs)",
                 epoch, epochs, train_loss, val_loss,
                 val_metrics["eer_percent"], val_metrics["roc_auc"],
                 time.perf_counter() - started)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_eer_percent": val_metrics["eer_percent"],
            "val_roc_auc": round(val_metrics["roc_auc"], 5),
        })

        if val_metrics["eer"] < best_eer:
            best_eer, best_metrics = val_metrics["eer"], val_metrics
            torch.save(model.state_dict(), weights_path)
            log.info("  saved best checkpoint (EER %.3f%%)", val_metrics["eer_percent"])
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("no improvement for %d epochs, stopping early", patience)
                break

    if not best_metrics:
        raise RuntimeError("training produced no valid validation metrics")

    model.load_state_dict(torch.load(weights_path, map_location=device,
                                     weights_only=True))
    return model, {"best": best_metrics, "history": history,
                   "weights_path": str(weights_path)}


@torch.no_grad()
def score_cache(model, cache, batch_size: int = 128):
    """Score a whole cached partition. Returns (labels, scores, attack ids)."""
    device = next(model.parameters()).device
    model.eval()
    ds = CachedDataset(cache)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, truths = [], []
    for inputs, targets, _ in loader:
        probs = torch.sigmoid(model(inputs.to(device))).cpu().numpy().ravel()
        scores.extend(probs.tolist())
        truths.extend(targets.numpy().ravel().tolist())
    attacks = cache.attacks[cache.valid_indices]
    return np.asarray(truths), np.asarray(scores), np.asarray(attacks)


@torch.no_grad()
def score_corpus(model, corpus, preprocessor, device, batch_size=32):
    """Score an entire corpus, returning (labels, scores, attack_ids)."""
    model.eval()
    ds = SpectrogramDataset(corpus, preprocessor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, truths = [], []
    for inputs, targets, valid in loader:
        probs = torch.sigmoid(model(inputs.to(device))).cpu().numpy().ravel()
        keep = valid.numpy().astype(bool)
        scores.extend(probs[keep].tolist())
        truths.extend(targets.numpy().ravel()[keep].tolist())
    return np.asarray(truths), np.asarray(scores), np.asarray(corpus.attack_ids)


def write_metadata(
    path: Path,
    corpus: datasets.Corpus,
    metrics: Dict,
    threshold: float,
    eval_metrics: Optional[Dict] = None,
    per_attack: Optional[Dict] = None,
    epochs: int = 0,
) -> None:
    """
    Record what this checkpoint actually is.

    This file is the provenance record the serving runtime trusts. It is the
    structural fix for the central defect in v1: a checkpoint trained on
    generated tones was indistinguishable, at load time, from one trained on
    ASVspoof.
    """
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_name": corpus.name,
        "training_data_kind": corpus.kind,
        "dataset_summary": corpus.summary(),
        "epochs_run": epochs,
        "decision_threshold": round(float(threshold), 6),
        "threshold_basis": "Equal Error Rate operating point on the validation split",
        "metrics": {"validation": metrics, "evaluation": eval_metrics or {}},
        "per_attack_eer": per_attack or {},
        "audio_frontend": {
            "sample_rate": config.SAMPLE_RATE,
            "window_seconds": config.WINDOW_SECONDS,
            "n_mels": config.N_MELS,
            "n_fft": config.N_FFT,
            "hop_length": config.HOP_LENGTH,
            "target_frames": config.TARGET_FRAMES,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    if corpus.kind != "real_corpus":
        meta["warning"] = (
            "Trained on generated signals, not speech. Not a detector. Do not "
            "quote these metrics as detection performance."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the synthetic-speech detector")
    parser.add_argument("--dataset", required=True,
                        choices=["asvspoof", "folder", "synthetic-demo"])
    parser.add_argument("--data-root", type=Path, default=None,
                        help="LA/ directory for asvspoof, or the corpus root for folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None,
                        help="stratified subsample size for quick runs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--roc-path", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_corpus: Optional[datasets.Corpus] = None

    if args.dataset == "asvspoof":
        if not args.data_root:
            parser.error("--data-root is required for --dataset asvspoof")
        train_corpus = datasets.asvspoof_2019_la(args.data_root, "train", args.limit, args.seed)
        val_corpus = datasets.asvspoof_2019_la(args.data_root, "dev", args.limit, args.seed)
        try:
            eval_corpus = datasets.asvspoof_2019_la(args.data_root, "eval", args.limit, args.seed)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("eval partition unavailable (%s); reporting dev only", exc)

    elif args.dataset == "folder":
        if not args.data_root:
            parser.error("--data-root is required for --dataset folder")
        corpus = datasets.folder_corpus(args.data_root)
        train_corpus, val_corpus = datasets.speaker_disjoint_split(corpus, 0.2, args.seed)

    else:
        log.warning("=" * 70)
        log.warning("SYNTHETIC DEMO MODE - generated tones, not speech.")
        log.warning("The resulting checkpoint is a pipeline smoke test only.")
        log.warning("=" * 70)
        root = Path(args.out_dir or ".") / "synthetic_demo_data"
        corpus = datasets.synthetic_demo(root, per_class=args.limit or 150, seed=args.seed)
        train_corpus, val_corpus = datasets.speaker_disjoint_split(corpus, 0.25, args.seed)

    model, result = train_model(
        train_corpus, val_corpus,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        num_workers=args.num_workers, out_dir=args.out_dir,
    )

    best = result["best"]
    threshold = best["eer_threshold"]
    print("\nVALIDATION")
    print(format_report(best))

    eval_metrics, attack_breakdown = None, None
    if eval_corpus is not None:
        device = next(model.parameters()).device
        y, s, attacks = score_corpus(model, eval_corpus, AudioPreprocessor(), device)
        eval_metrics = evaluate_scores(y, s, threshold=threshold)
        attack_breakdown = per_attack_eer(y, s, attacks)
        print("\nEVALUATION (held-out partition, threshold fixed from validation)")
        print(format_report(eval_metrics, attack_breakdown))

        roc_path = args.roc_path or (Path(result["weights_path"]).parent / "roc_curve.png")
        save_roc_curve(y, s, str(roc_path),
                       title=f"{eval_corpus.name} - ROC")
        log.info("wrote %s", roc_path)

    meta_path = Path(result["weights_path"]).parent / "model_meta.json"
    write_metadata(
        meta_path, train_corpus, best, threshold,
        eval_metrics, attack_breakdown, epochs=len(result["history"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
