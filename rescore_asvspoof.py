"""
Re-score the trained checkpoint against the cached partitions.

Scoring is cheap once the feature caches exist, so metric changes and reporting
fixes do not require retraining. Loads the saved weights, re-runs the
development and evaluation partitions, and rewrites the metrics report, the
provenance record and the ROC figure.

    python rescore_asvspoof.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model.cache import FeatureCache
from model.cnn_architecture import VoiceDeepfakeCNN
from model.evaluate import (evaluate_scores, format_report, per_attack_eer,
                            save_roc_curve)
from model.train import score_cache, write_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rescore")

CACHE = Path("D:/Before_Reset/Documents/SLIIT/Y3S2/ISP_Assignment/datasets/cache")
OUT = Path("model/saved_models")


def main() -> int:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(OUT / "voice_deepfake_cnn.pth",
                                     map_location=device, weights_only=True))
    model.eval()

    caches = {p: FeatureCache(CACHE / p) for p in ("train", "dev", "eval")}
    for name, cache in caches.items():
        log.info("%s: %s", name, json.dumps(cache.summary()))

    # Development partition sets the operating point.
    y_dev, s_dev, _ = score_cache(model, caches["dev"])
    dev = evaluate_scores(y_dev, s_dev)
    threshold = dev["eer_threshold"]
    print("\nDEVELOPMENT PARTITION (A01-A06, seen during training)")
    print(format_report(dev))

    # Evaluation partition is scored at that same threshold, unchanged.
    y_ev, s_ev, attacks = score_cache(model, caches["eval"])
    ev = evaluate_scores(y_ev, s_ev, threshold=threshold)
    ev_at_own_eer = evaluate_scores(y_ev, s_ev)
    breakdown = per_attack_eer(y_ev, s_ev, attacks)

    print("\nEVALUATION PARTITION (A07-A19, never seen; threshold fixed from dev)")
    print(format_report(ev, breakdown))

    roc = Path("docs/roc_curve_asvspoof.png")
    roc.parent.mkdir(parents=True, exist_ok=True)
    save_roc_curve(y_ev, s_ev, str(roc),
                   title="ASVspoof 2019 LA evaluation - ROC")

    train_summary = caches["train"].summary()

    class _Meta:
        name = "ASVspoof2019-LA-train"
        kind = "real_corpus"

        def summary(self_inner):
            return train_summary

    write_metadata(OUT / "model_meta.json", _Meta(), dev, threshold,
                   ev, breakdown, epochs=16)

    previous = {}
    report_path = Path("docs/training_report.json")
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))

    report = {
        "dataset": "ASVspoof 2019 Logical Access (official distribution)",
        "partitions": {k: v.summary() for k, v in caches.items()},
        "speaker_overlap_with_train": previous.get("speaker_overlap_with_train", 0),
        "development": dev,
        "evaluation_at_dev_threshold": ev,
        "evaluation_at_own_eer": ev_at_own_eer,
        "per_attack_eer": breakdown,
        "history": previous.get("history", []),
        "rescored_minutes": round((time.time() - started) / 60, 1),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s and %s", report_path, OUT / "model_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
