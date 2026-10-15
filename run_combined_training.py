"""
Retrain the detector on real voices from many recording conditions.

Why: the first model was trained only on ASVspoof 2019 LA, whose genuine half
is one studio corpus. It learned that studio speech (and the silence around
it) means "real", so a real voice recorded on an ordinary microphone was often
flagged as fake. This run fixes that in four ways:

  1. Silence is trimmed in training and inference (shared preprocessor).
  2. LibriSpeech (home recordings, many speakers and microphones) is added as
     genuine speech, speaker-disjoint between training, validation and test.
  3. Each LibriSpeech clip also gets a vocoder-resynthesised twin labelled as
     spoofed, so "sounds like LibriSpeech" cannot become a new shortcut.
  4. Half of all training clips pass through a random channel (phone line,
     G.711, noise, low-pass), for genuine and spoofed clips alike.

Model selection weighs the two domains equally, so the 24,844 ASVspoof
development clips cannot drown out the real-voice check.

    python run_combined_training.py
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model.augment import random_kind
from model.cache import FeatureCache
from model.cnn_architecture import VoiceDeepfakeCNN, count_parameters
from model.evaluate import evaluate_scores, per_attack_eer, save_roc_curve
from model.jobs import LA_ZIP, build_job_cache
from model.train import CachedDataset, _run_epoch, score_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("combined")

DATA = Path("D:/Before_Reset/Documents/SLIIT/Y3S2/ISP_Assignment/datasets")
LIBRI = DATA / "librispeech" / "LibriSpeech"
CACHE = DATA / "cache_v2"
OLD_CACHE = DATA / "cache"
OUT = Path("model/saved_models")
PROTO = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
         "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}


# ------------------------------------------------------------------ data lists
def asv_rows(zf, part):
    text = zf.read(f"LA/ASVspoof2019_LA_cm_protocols/{PROTO[part]}").decode()
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append((f"LA/ASVspoof2019_LA_{part}/flac/{p[1]}.flac", p[0],
                         "bonafide" if p[4] == "bonafide" else p[3],
                         0.0 if p[4] == "bonafide" else 1.0))
    return rows


def libri_files(subsets):
    out = []
    for s in subsets:
        for f in sorted((LIBRI / s).rglob("*.flac")):
            out.append((str(f), f.name.split("-")[0]))
    return out


def weighted_eer(labels, scores, weights):
    """EER with per-sample weights, so each domain counts equally."""
    fpr, tpr, thr = roc_curve(labels, scores, sample_weight=weights)
    keep = np.isfinite(thr)
    fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def balanced(caches_scores):
    """Concatenate (labels, scores) from several domains with equal total weight each."""
    ys, ss, ws = [], [], []
    for y, s in caches_scores:
        ys.append(y); ss.append(s); ws.append(np.full(len(y), 1.0 / len(y)))
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(ws)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    zf = zipfile.ZipFile(LA_ZIP)
    W = args.workers

    # keep the first model for the before/after comparison
    v1w, v1m = OUT / "voice_deepfake_cnn_v1_asvspoof_only.pth", OUT / "model_meta_v1.json"
    if not v1w.exists():
        shutil.copy2(OUT / "voice_deepfake_cnn.pth", v1w)
        shutil.copy2(OUT / "model_meta.json", v1m)
    old_thr = json.loads(v1m.read_text(encoding="utf-8"))["decision_threshold"]

    # ---- LibriSpeech speaker split: dev-* -> train/val (80/20 by speaker), test-* -> test
    dev_files = libri_files(["dev-clean", "dev-other"])
    test_files = libri_files(["test-clean", "test-other"])
    speakers = sorted({s for _, s in dev_files})
    rng.shuffle(speakers)
    val_spk = set(speakers[: max(1, len(speakers) // 5)])
    lib_train = [f for f in dev_files if f[1] not in val_spk]
    lib_val = [f for f in dev_files if f[1] in val_spk]
    assert not ({s for _, s in lib_train} & {s for _, s in test_files}), "speaker leak"
    log.info("LibriSpeech speakers: train %d, val %d, test %d",
             len({s for _, s in lib_train}), len(val_spk), len({s for _, s in test_files}))

    def jobs_for(items, kind_fn, voc_fn, tag_fn, label_fn, src_kind):
        jobs, labels, tags = [], [], []
        for i, it in enumerate(items):
            jobs.append((i, src_kind, it[0], voc_fn(i, it), kind_fn(i, it), int(rng.integers(1 << 30)), False))
            labels.append(label_fn(i, it)); tags.append(tag_fn(i, it))
        return jobs, labels, tags

    aug = lambda i, it: (lambda k: None if k == "clean" else k)(random_kind(rng))  # noqa: E731
    none = lambda i, it: None  # noqa: E731

    # ---- training caches
    asv_tr = asv_rows(zf, "train")
    c_asv_tr = build_job_cache("asv_train", *jobs_for(asv_tr, aug, none, lambda i, r: r[2], lambda i, r: r[3], "zip"),
                               CACHE / "asv_train", W)
    lib_tr_items = lib_train + lib_train                         # real copy, then vocoded copy
    n_real = len(lib_train)
    c_lib_tr = build_job_cache("libri_train", *jobs_for(
        lib_tr_items, aug, lambda i, it: None if i < n_real else int(i % 2),
        lambda i, it: "libri_real" if i < n_real else f"libri_vocoded{i % 2}",
        lambda i, it: 0.0 if i < n_real else 1.0, "file"), CACHE / "libri_train", W)

    # ---- validation caches (clean)
    asv_dev = asv_rows(zf, "dev")
    by_att = defaultdict(list)
    for r in asv_dev:
        by_att[r[2]].append(r)
    dev_pick = list(by_att["bonafide"])
    for a, rows in sorted(by_att.items()):
        if a != "bonafide":
            idx = rng.choice(len(rows), size=min(667, len(rows)), replace=False)
            dev_pick += [rows[j] for j in idx]
    c_asv_val = build_job_cache("asv_val", *jobs_for(dev_pick, none, none, lambda i, r: r[2], lambda i, r: r[3], "zip"),
                                CACHE / "asv_val", W)
    lib_val_items = lib_val + lib_val
    nv = len(lib_val)
    c_lib_val = build_job_cache("libri_val", *jobs_for(
        lib_val_items, none, lambda i, it: None if i < nv else int(i % 2),
        lambda i, it: "libri_real" if i < nv else f"libri_vocoded{i % 2}",
        lambda i, it: 0.0 if i < nv else 1.0, "file"),
        CACHE / "libri_val", W)

    # ---- test caches (clean)
    asv_ev = asv_rows(zf, "eval")
    c_asv_ev = build_job_cache("asv_eval", *jobs_for(asv_ev, none, none, lambda i, r: r[2], lambda i, r: r[3], "zip"),
                               CACHE / "asv_eval", W)
    lib_test_items = test_files + test_files
    nt = len(test_files)
    c_lib_te = build_job_cache("libri_test", *jobs_for(
        lib_test_items, none, lambda i, it: None if i < nt else int(i % 2),
        lambda i, it: "libri_real" if i < nt else f"libri_vocoded{i % 2}",
        lambda i, it: 0.0 if i < nt else 1.0, "file"),
        CACHE / "libri_test", W)
    # old front end on the same real test voices, for the "before" figure
    legacy_jobs = [(i, "file", f, None, None, 0, True) for i, (f, _) in enumerate(test_files)]
    c_lib_legacy = build_job_cache("libri_test_legacy", legacy_jobs, [0.0] * nt, ["libri_real"] * nt,
                                   CACHE / "libri_test_legacy", W)
    log.info("caches ready after %.1f min", (time.time() - t0) / 60)

    # ---- train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = ConcatDataset([CachedDataset(c_asv_tr), CachedDataset(c_lib_tr)])
    n_b = sum(CachedDataset(c).class_counts[0] for c in (c_asv_tr, c_lib_tr))
    n_s = sum(CachedDataset(c).class_counts[1] for c in (c_asv_tr, c_lib_tr))
    log.info("training clips: %d (genuine %d, spoofed %d)", len(ds_tr), n_b, n_s)
    loader = DataLoader(ds_tr, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(ConcatDataset([CachedDataset(c_asv_val), CachedDataset(c_lib_val)]),
                            batch_size=128)

    model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_b / n_s], device=device), reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    best, stale, history = float("inf"), 0, []
    wpath = OUT / "voice_deepfake_cnn.pth"

    for ep in range(1, args.epochs + 1):
        st = time.time()
        tl, _, _ = _run_epoch(model, loader, crit, opt, device, True)
        vl, _, _ = _run_epoch(model, val_loader, crit, opt, device, False)
        sched.step(vl)
        ya, sa, _ = score_cache(model, c_asv_val)
        yl, sl, _ = score_cache(model, c_lib_val)
        e_asv = evaluate_scores(ya, sa)["eer_percent"]
        e_lib = evaluate_scores(yl, sl)["eer_percent"]
        e_bal, _ = weighted_eer(*balanced([(ya, sa), (yl, sl)]))
        history.append({"epoch": ep, "train_loss": round(tl, 5), "val_loss": round(vl, 5),
                        "val_eer_asvspoof": e_asv, "val_eer_librispeech": e_lib,
                        "val_eer_balanced": round(e_bal * 100, 3)})
        log.info("epoch %2d  loss %.4f/%.4f  EER asv %.2f%%  libri %.2f%%  balanced %.2f%%  (%.0fs)",
                 ep, tl, vl, e_asv, e_lib, e_bal * 100, time.time() - st)
        if e_bal < best:
            best, stale = e_bal, 0
            torch.save(model.state_dict(), wpath)
            log.info("  saved best (balanced EER %.3f%%)", e_bal * 100)
        else:
            stale += 1
            if stale >= args.patience:
                log.info("no improvement for %d epochs, stopping", args.patience)
                break

    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
    model.eval()
    ya, sa, _ = score_cache(model, c_asv_val)
    yl, sl, _ = score_cache(model, c_lib_val)
    val_bal_eer, thr = weighted_eer(*balanced([(ya, sa), (yl, sl)]))
    log.info("decision threshold (balanced EER point): %.4f", thr)

    # ---- evaluate new model
    y_ae, s_ae, a_ae = score_cache(model, c_asv_ev)
    y_lt, s_lt, _ = score_cache(model, c_lib_te)
    new = {
        "asvspoof_eval": evaluate_scores(y_ae, s_ae, threshold=thr),
        "asvspoof_eval_per_attack": per_attack_eer(y_ae, s_ae, a_ae),
        "librispeech_test": evaluate_scores(y_lt, s_lt, threshold=thr),
        "librispeech_real_flagged_percent": round(float(np.mean(s_lt[y_lt == 0] >= thr)) * 100, 2),
    }
    save_roc_curve(y_ae, s_ae, "docs/figures/roc_curve_v2.png", title="ASVspoof 2019 LA evaluation - ROC (v2)")

    # ---- evaluate old model the way the old service ran it
    old = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE)
    old.load_state_dict(torch.load(v1w, map_location="cpu", weights_only=True)); old.to(device).eval()
    y_oe, s_oe, _ = score_cache(old, FeatureCache(OLD_CACHE / "eval"))
    _, s_ol, _ = score_cache(old, c_lib_legacy)
    y_ot, s_ot, _ = score_cache(old, c_asv_ev)
    before = {
        "asvspoof_eval_eer_percent": evaluate_scores(y_oe, s_oe)["eer_percent"],
        "asvspoof_eval_silence_trimmed_eer_percent": evaluate_scores(y_ot, s_ot)["eer_percent"],
        "librispeech_real_flagged_percent": round(float(np.mean(s_ol >= old_thr)) * 100, 2),
        "threshold": old_thr,
    }

    # ---- robustness: same stratified design as the earlier readiness test
    rob = {}
    ev_by = defaultdict(list)
    for r in asv_ev:
        ev_by[r[2]].append(r)
    r2 = np.random.default_rng(7)
    rob_asv = []
    for a, rows in sorted(ev_by.items()):
        k = 600 if a == "bonafide" else 80
        rob_asv += [rows[j] for j in r2.choice(len(rows), size=k, replace=False)]
    rob_lib = [test_files[j] for j in r2.choice(nt, size=300, replace=False)]
    for cond in ("clean", "telephone", "g711", "noise20"):
        ch = None if cond == "clean" else cond
        ja = [(i, "zip", r[0], None, ch, 11 + i, False) for i, r in enumerate(rob_asv)]
        ca = build_job_cache(f"rob_asv_{cond}", ja, [r[3] for r in rob_asv], [r[2] for r in rob_asv],
                             CACHE / f"rob_asv_{cond}", W)
        jl = [(i, "file", f, None, ch, 99 + i, False) for i, (f, _) in enumerate(rob_lib)]
        cl = build_job_cache(f"rob_lib_{cond}", jl, [0.0] * len(rob_lib), ["libri_real"] * len(rob_lib),
                             CACHE / f"rob_lib_{cond}", W)
        y, s, _ = score_cache(model, ca)
        _, sr, _ = score_cache(model, cl)
        m = evaluate_scores(y, s, threshold=thr)
        rob[cond] = {"asvspoof_eer_percent": m["eer_percent"],
                     "clones_accepted_percent": round(m["false_acceptance_rate"] * 100, 2),
                     "asvspoof_genuine_refused_percent": round(m["false_rejection_rate"] * 100, 2),
                     "librispeech_real_flagged_percent": round(float(np.mean(sr >= thr)) * 100, 2)}
        log.info("robustness %-9s %s", cond, rob[cond])

    # ---- write artefacts
    summary = {
        "asvspoof_train": c_asv_tr.summary(), "librispeech_train": c_lib_tr.summary(),
        "librispeech_speakers": {"train": len({s for _, s in lib_train}), "val": len(val_spk),
                                 "test": len({s for _, s in test_files})},
        "training_clips": len(ds_tr), "genuine": n_b, "spoofed": n_s,
    }
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_name": "ASVspoof2019-LA-train + LibriSpeech dev (real and vocoded)",
        "training_data_kind": "real_corpus",
        "dataset_summary": summary,
        "epochs_run": len(history),
        "decision_threshold": round(thr, 6),
        "threshold_basis": "EER point on validation, ASVspoof and LibriSpeech weighted equally",
        "preprocessing": {"silence_trim_top_db": config.TRIM_TOP_DB, "short_clip_padding": "repeat",
                          "channel_augmentation": "50% of training clips"},
        "metrics": {"validation": {"balanced_eer_percent": round(val_bal_eer * 100, 3)},
                    "evaluation": new["asvspoof_eval"]},
        "per_attack_eer": new["asvspoof_eval_per_attack"],
        "audio_frontend": {"sample_rate": config.SAMPLE_RATE, "window_seconds": config.WINDOW_SECONDS,
                           "n_mels": config.N_MELS, "n_fft": config.N_FFT,
                           "hop_length": config.HOP_LENGTH, "target_frames": config.TARGET_FRAMES},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = {"summary": summary, "history": history, "threshold": thr, "new_model": new,
              "old_model": before, "robustness": rob, "minutes": round((time.time() - t0) / 60, 1)}
    Path("docs/training_report_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("BEFORE: %s", before)
    log.info("AFTER : asv EER %.3f%% | libri EER %.3f%% | real voices flagged %.2f%%",
             new["asvspoof_eval"]["eer_percent"], new["librispeech_test"]["eer_percent"],
             new["librispeech_real_flagged_percent"])
    log.info("total %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
