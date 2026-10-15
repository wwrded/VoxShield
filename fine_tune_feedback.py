"""
Learn from operator corrections, safely.

Operators send back clips the model got wrong, with the right answer
(POST /api/feedback, or the "Was this result right?" box on the dashboard).
This script turns that queue into a small, checked model update:

1. Each correction becomes several training examples: different 3-second
   crops of the clip, some passed through the same phone, codec and noise
   effects used in main training. One clip alone is too little to learn from.
2. The corrections are mixed with a random sample of the original training
   data ("replay"). Fine-tuning on a handful of new clips alone makes a network
   forget what it knew; replay keeps the old knowledge in view.
3. Training runs for a few epochs at a low learning rate, starting from the
   current model, with batch-norm statistics frozen so a small set cannot
   shift every prediction at once.
4. A gate decides whether the update is installed. The new model must be no
   worse than the old one on the held-out ASVspoof and LibriSpeech validation
   sets (within --max-drop percentage points of balanced EER), and it must get
   at least as many of the corrections right as the old model did. A poisoned
   or mislabelled batch that would damage the detector fails the first test.
5. If installed, the old weights are kept as a backup, the provenance record
   notes the fine-tune, and the used corrections move out of the queue.

    python fine_tune_feedback.py              # fine-tune and install if it passes
    python fine_tune_feedback.py --dry-run    # report only, change nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from backend import feedback
from backend.preprocessor import AudioPreprocessor
from model import augment
from model.cache import FeatureCache
from model.cnn_architecture import VoiceDeepfakeCNN
from model.train import CachedDataset, score_cache

CACHE = Path("D:/Before_Reset/Documents/SLIIT/Y3S2/ISP_Assignment/datasets/cache_v2")
OUT = Path(config.WEIGHTS_PATH).parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("fine_tune")


# ------------------------------------------------------------------ metrics
def balanced_eer(pairs):
    """EER over several domains, each weighted equally. Returns (eer, threshold)."""
    ys, ss, ws = [], [], []
    for y, s in pairs:
        ys.append(y); ss.append(s); ws.append(np.full(len(y), 1.0 / len(y)))
    fpr, tpr, thr = roc_curve(np.concatenate(ys), np.concatenate(ss),
                              sample_weight=np.concatenate(ws))
    keep = np.isfinite(thr)
    fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    i = int(np.argmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def accept_update(old_eer: float, new_eer: float, old_right: int, new_right: int,
                  max_drop: float) -> tuple:
    """The installation gate. Returns (accepted, reason)."""
    if new_eer > old_eer + max_drop / 100.0:
        return False, (f"validation EER rose from {old_eer:.2%} to {new_eer:.2%}, "
                       f"more than the {max_drop} point limit")
    if new_right < old_right:
        return False, f"it gets fewer corrections right ({new_right} vs {old_right})"
    return True, "no worse on held-out data and at least as good on the corrections"


# ------------------------------------------------------------------ data
class TensorSet(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], torch.tensor(1.0)


def correction_examples(items, pre, copies, rng):
    """Several crops and channel variants per corrected clip."""
    xs, ys, centre_x, centre_y = [], [], [], []
    n = pre.window_samples
    for it in items:
        try:
            y = pre.trim_silence(pre.load_waveform(it["path"]))
        except ValueError as exc:
            log.warning("skipping %s: %s", it["id"], exc)
            continue
        label = feedback.LABELS[it["label"]]
        centre_x.append(pre.mel_spectrogram(pre.fit_length(y, n))[None])
        centre_y.append(label)
        for k in range(copies):
            if len(y) > n:
                s = int(rng.integers(0, len(y) - n + 1))
                w = y[s:s + n]
            else:
                w = pre.fit_length(y, n)
            kind = "clean" if k == 0 else augment.random_kind(rng)
            w = np.asarray(augment.apply(w, kind, rng), dtype=np.float32)
            peak = float(np.max(np.abs(w)))
            if peak > 1e-4:
                w = w / peak
            xs.append(pre.mel_spectrogram(pre.fit_length(w, n))[None])
            ys.append(label)
    as_arr = lambda a: np.stack(a) if a else np.zeros((0, 1, config.N_MELS, config.TARGET_FRAMES))
    return as_arr(xs), np.array(ys), as_arr(centre_x), np.array(centre_y)


def score_array(model, x):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x.astype(np.float32)))).numpy().ravel()


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--replay", type=int, default=4000, help="original clips mixed in")
    ap.add_argument("--copies", type=int, default=24, help="training variants per correction")
    ap.add_argument("--max-drop", type=float, default=0.5,
                    help="allowed rise in validation balanced EER, percentage points")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    items = feedback.load_pending()
    if not items:
        log.info("no corrections in the queue (%s); nothing to do", feedback.pending_dir())
        return 0
    counts = {k: sum(i["label"] == k for i in items) for k in feedback.LABELS}
    log.info("corrections: %d (%s)", len(items), counts)

    meta = json.loads(Path(config.META_PATH).read_text(encoding="utf-8"))
    old_thr = float(meta["decision_threshold"])
    model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE)
    model.load_state_dict(torch.load(config.WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()

    pre = AudioPreprocessor()
    fx, fy, cx, cy = correction_examples(items, pre, args.copies, rng)
    if not len(cx):
        log.error("no correction could be decoded")
        return 1

    val = [FeatureCache(CACHE / "asv_val"), FeatureCache(CACHE / "libri_val")]

    def evaluate(m, thr=None):
        m.eval()
        pairs = [score_cache(m, c)[:2] for c in val]
        eer, t = balanced_eer(pairs)
        t = t if thr is None else thr
        right = int(np.sum((score_array(m, cx) >= t) == (cy == 1)))
        return eer, t, right

    old_eer, _, old_right = evaluate(model, old_thr)
    log.info("current model: validation balanced EER %.3f%%, corrections right %d/%d",
             old_eer * 100, old_right, len(cy))

    # replay: a balanced random sample of the original training caches
    replay = []
    for name in ("asv_train", "libri_train"):
        ds = CachedDataset(FeatureCache(CACHE / name))
        k = min(args.replay // 2, len(ds))
        replay.append(Subset(ds, rng.choice(len(ds), size=k, replace=False).tolist()))
    train_set = ConcatDataset(replay + [TensorSet(fx, fy)])
    loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)

    labels = np.concatenate(
        [s.dataset.labels[s.dataset.index[s.indices]] for s in replay] + [fy])
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep the normalisation statistics learned from 34,204 clips; a small
        # fine-tuning set that overwrote them would shift every prediction.
        for layer in model.modules():
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                layer.eval()
        total, seen = 0.0, 0
        for x, y, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach()) * len(x); seen += len(x)
        log.info("epoch %d  loss %.4f", epoch, total / max(seen, 1))

    new_eer, new_thr, new_right = evaluate(model)
    log.info("fine-tuned model: validation balanced EER %.3f%%, threshold %.4f, "
             "corrections right %d/%d", new_eer * 100, new_thr, new_right, len(cy))

    ok, reason = accept_update(old_eer, new_eer, old_right, new_right, args.max_drop)
    log.info("%s: %s", "ACCEPTED" if ok else "REJECTED", reason)
    if not ok or args.dry_run:
        if args.dry_run:
            log.info("dry run: nothing installed, queue unchanged")
        return 0 if ok else 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(config.WEIGHTS_PATH, OUT / f"voice_deepfake_cnn_before_feedback_{stamp}.pth")
    shutil.copy2(config.META_PATH, OUT / f"model_meta_before_feedback_{stamp}.json")
    torch.save(model.state_dict(), config.WEIGHTS_PATH)

    meta["decision_threshold"] = new_thr
    meta.setdefault("fine_tunes", []).append({
        "at": stamp,
        "corrections": len(cy),
        "labels": counts,
        "validation_balanced_eer_percent": {"before": round(old_eer * 100, 3),
                                            "after": round(new_eer * 100, 3)},
        "corrections_right": {"before": old_right, "after": new_right},
    })
    Path(config.META_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    used = Path(config.FEEDBACK_DIR) / "used" / stamp
    used.mkdir(parents=True, exist_ok=True)
    for it in items:
        for p in (Path(it["path"]), feedback.pending_dir() / f"{it['id']}.json"):
            if p.exists():
                shutil.move(str(p), str(used / p.name))
    log.info("installed. Restart the service to load it. Backup: %s", stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
