"""
Deployment-readiness check: how does the trained detector hold up under the
channel conditions a real deployment would feed it?

ASVspoof 2019 LA is clean studio audio. A bank's phone line is not. This script
takes a stratified sample of the held-out evaluation partition, re-scores it
under four conditions, and reports the Equal Error Rate for each:

  clean      the audio as distributed (should match the reported 11.668 %)
  telephone  band-limited to 300-3400 Hz at 8 kHz, as on a narrowband call
  g711       8 kHz, 8-bit mu-law companding - the codec most landlines use
  noise      white noise added at 20 dB signal-to-noise ratio

    python evaluate_robustness.py [--per-attack 80] [--bonafide 600]

Audio is read straight out of LA.zip, so the corpus need not be extracted.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from backend.preprocessor import AudioPreprocessor
from model.cnn_architecture import VoiceDeepfakeCNN
from model.evaluate import compute_eer

ZIP = Path("D:/Before_Reset/Documents/SLIIT/Y3S2/ISP_Assignment/datasets/LA.zip")
PROTOCOL = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO = "LA/ASVspoof2019_LA_eval/flac/{}.flac"
SR = config.SAMPLE_RATE


# ------------------------------------------------------------------ channels
# Shared with training so the test conditions and the augmentation are identical code.
from model import augment

CONDITIONS = {
    "clean": lambda y: y,
    "telephone": augment.telephone,
    "g711": augment.g711,
    "noise_20db": lambda y, rng=np.random.default_rng(0): augment.add_noise(y, 20.0, rng),
}


# ------------------------------------------------------------------ sampling
def sample_protocol(zf, per_attack: int, n_bonafide: int, seed: int = 7):
    rows = zf.read(PROTOCOL).decode().splitlines()
    by_class = defaultdict(list)
    for line in rows:
        parts = line.split()
        if len(parts) >= 5:
            key = "bonafide" if parts[4] == "bonafide" else parts[3]
            by_class[key].append(parts[1])
    rng = np.random.default_rng(seed)
    chosen = []
    for key, utts in sorted(by_class.items()):
        k = n_bonafide if key == "bonafide" else per_attack
        pick = rng.choice(utts, size=min(k, len(utts)), replace=False)
        chosen += [(u, key) for u in pick]
    return chosen


def centre_window(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)))
    s = (len(y) - n) // 2
    return y[s:s + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-attack", type=int, default=80)
    ap.add_argument("--bonafide", type=int, default=600)
    args = ap.parse_args()

    t0 = time.time()
    pre = AudioPreprocessor()
    model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE)
    model.load_state_dict(torch.load("model/saved_models/voice_deepfake_cnn.pth",
                                     map_location="cpu", weights_only=True))
    model.eval()
    meta = json.loads(Path("model/saved_models/model_meta.json").read_text(encoding="utf-8"))
    threshold = float(meta["decision_threshold"])

    zf = zipfile.ZipFile(ZIP)
    sample = sample_protocol(zf, args.per_attack, args.bonafide)
    print(f"sample: {len(sample)} utterances "
          f"({sum(1 for _, k in sample if k == 'bonafide')} bonafide)", flush=True)

    feats = {c: [] for c in CONDITIONS}
    labels, attacks = [], []
    for i, (utt, key) in enumerate(sample):
        y = pre.trim_silence(pre.load_waveform(io.BytesIO(zf.read(AUDIO.format(utt)))))
        for cond, fn in CONDITIONS.items():
            z = np.asarray(fn(y), dtype=np.float32)
            peak = float(np.max(np.abs(z)))
            if peak > 1e-4:
                z = z / peak
            feats[cond].append(pre.mel_spectrogram(pre.fit_length(z, pre.window_samples)))
        labels.append(0 if key == "bonafide" else 1)
        attacks.append(key)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(sample)}", flush=True)

    labels = np.array(labels)
    attacks = np.array(attacks)
    results = {}
    for cond, specs in feats.items():
        x = torch.from_numpy(np.stack(specs)[:, None])
        with torch.no_grad():
            scores = torch.cat([model.predict_proba(x[i:i + 256])
                                for i in range(0, len(x), 256)]).numpy()
        eer, _ = compute_eer(labels, scores)
        clone_accept = float(np.mean(scores[labels == 1] < threshold))
        genuine_refused = float(np.mean(scores[labels == 0] >= threshold))
        worst = {}
        for a in sorted(set(attacks) - {"bonafide"}):
            m = (attacks == a) | (labels == 0)
            worst[a] = round(compute_eer(labels[m], scores[m])[0] * 100, 2)
        results[cond] = {
            "eer_percent": round(eer * 100, 3),
            "at_deployed_threshold": {
                "clones_accepted_percent": round(clone_accept * 100, 2),
                "genuine_refused_percent": round(genuine_refused * 100, 2),
            },
            "per_attack_eer_percent": worst,
        }
        print(f"{cond:<11} EER {eer*100:6.2f}%   at deployed threshold: "
              f"clones accepted {clone_accept*100:5.1f}%  genuine refused "
              f"{genuine_refused*100:5.1f}%", flush=True)

    report = {
        "sample_size": int(len(labels)),
        "bonafide": int((labels == 0).sum()),
        "spoof": int((labels == 1).sum()),
        "deployed_threshold": threshold,
        "conditions": results,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    Path("docs/robustness_report.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print("wrote docs/robustness_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
