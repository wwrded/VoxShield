"""
End-to-end training and evaluation on ASVspoof 2019 Logical Access.

    python run_asvspoof_training.py --zip datasets/LA.zip --epochs 20

Reads the official protocol files and audio straight out of the distributed
archive, caches features once, trains, then evaluates on the held-out
evaluation partition using the threshold calibrated on the development
partition. Writes the weights, a provenance record, a metrics report and a ROC
figure.

The evaluation partition contains spoofing systems A07-A19, none of which
appear in training (A01-A06). The evaluation EER is therefore expected to be
substantially worse than the development EER: that gap is the result worth
reporting, because it measures generalisation to synthesis systems the model
has never seen, which is the situation any deployed detector is in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import asvspoof_zip
from model.evaluate import (evaluate_scores, format_report, per_attack_eer,
                            save_roc_curve)
from model.train import score_cache, train_from_caches, write_metadata

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("asvspoof")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path,
                    default=Path("datasets/LA.zip"))
    ap.add_argument("--cache-dir", type=Path, default=Path("datasets/cache"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("model/saved_models"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.zip.exists():
        log.error("archive not found: %s", args.zip)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()

    # ---------------------------------------------------------------- data
    limits = {"train": args.limit_train, "dev": args.limit_dev,
              "eval": args.limit_eval}
    caches = {}
    for part in ("train", "dev", "eval"):
        corpus = asvspoof_zip.load_partition(args.zip, part,
                                             limit=limits[part], seed=args.seed)
        log.info("%s protocol: %s", part, json.dumps(corpus.summary()))
        caches[part] = asvspoof_zip.build_zip_cache(
            args.zip, corpus, args.cache_dir / part,
            workers=args.workers, rebuild=args.rebuild_cache)

    # Speaker disjointness is a property of the official partitions; confirm it
    # rather than assume it, because a shared speaker would let the model pass
    # by recognising voices instead of detecting synthesis.
    spk = {}
    for part in ("train", "dev", "eval"):
        meta = json.loads((args.cache_dir / part / "meta.json").read_text(encoding="utf-8"))
        spk[part] = set(meta.get("speakers", []))
    overlap = (spk["train"] & spk["dev"]) | (spk["train"] & spk["eval"])
    log.info("speakers  train=%d dev=%d eval=%d  overlap with train=%d",
             len(spk["train"]), len(spk["dev"]), len(spk["eval"]), len(overlap))

    # ------------------------------------------------------------ training
    model, result = train_from_caches(
        caches["train"], caches["dev"],
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, out_dir=args.out_dir)

    best = result["best"]
    threshold = best["eer_threshold"]
    print("\nDEVELOPMENT PARTITION (A01-A06, seen during training)")
    print(format_report(best))

    # ---------------------------------------------------------- evaluation
    log.info("scoring the held-out evaluation partition")
    y, s, attacks = score_cache(model, caches["eval"])
    eval_metrics = evaluate_scores(y, s, threshold=threshold)
    breakdown = per_attack_eer(y, s, attacks)

    print("\nEVALUATION PARTITION (A07-A19, never seen; threshold fixed from dev)")
    print(format_report(eval_metrics, breakdown))

    # ------------------------------------------------------------ artefacts
    out = Path(result["weights_path"]).parent
    roc = out.parent.parent / "docs" / "roc_curve_asvspoof.png"
    roc.parent.mkdir(parents=True, exist_ok=True)
    save_roc_curve(y, s, str(roc),
                   title="ASVspoof 2019 LA evaluation - ROC")
    log.info("wrote %s", roc)

    train_corpus_summary = caches["train"].summary()

    class _Meta:
        name = "ASVspoof2019-LA-train"
        kind = "real_corpus"

        def summary(self_inner):
            return train_corpus_summary

    write_metadata(out / "model_meta.json", _Meta(), best, threshold,
                   eval_metrics, breakdown, epochs=len(result["history"]))

    report = {
        "dataset": "ASVspoof 2019 Logical Access (official distribution)",
        "partitions": {k: v.summary() for k, v in caches.items()},
        "speaker_overlap_with_train": len(overlap),
        "development": best,
        "evaluation": eval_metrics,
        "per_attack_eer": breakdown,
        "history": result["history"],
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    report_path = out.parent.parent / "docs" / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", report_path)
    log.info("total elapsed: %.1f min", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
