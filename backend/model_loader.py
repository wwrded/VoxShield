"""
Inference runtime: checkpoint loading, provenance, and windowed scoring.

The central design rule here is that the service must never present a number
as a model prediction unless a model actually produced it, and must never
present a model prediction as trustworthy unless the checkpoint says what it
was trained on. v1 silently fell back to a hand-tuned acoustic heuristic when
no weights were present and returned it in the same field, with the same
formatting, as a real prediction. That is the kind of defect that turns into a
false assurance in an operational setting, so it is fixed structurally: the
scorer that produced each number is named in the response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import config
from model.cnn_architecture import VoiceDeepfakeCNN, count_parameters

log = logging.getLogger(__name__)


class ModelStatus:
    """How much trust the loaded artefact has earned."""

    TRAINED = "TRAINED"                    # Real speech corpus, metrics recorded.
    DEMO_SYNTHETIC = "DEMO_SYNTHETIC"      # Trained on generated tones. Toy only.
    UNTRAINED = "UNTRAINED"                # Randomly initialised weights.
    HEURISTIC_ONLY = "HEURISTIC_ONLY"      # No checkpoint; signal-processing baseline.


UNTRUSTWORTHY = {
    ModelStatus.DEMO_SYNTHETIC,
    ModelStatus.UNTRAINED,
    ModelStatus.HEURISTIC_ONLY,
}

CAVEAT_TEXT = {
    ModelStatus.DEMO_SYNTHETIC: (
        "The loaded checkpoint was trained on procedurally generated tones, not "
        "on recorded human or synthesised speech. Scores demonstrate that the "
        "pipeline runs end to end; they say nothing about real-world accuracy "
        "and must not be quoted as detection performance."
    ),
    ModelStatus.UNTRAINED: (
        "No trained checkpoint is loaded. The network holds randomly "
        "initialised weights and its output is meaningless."
    ),
    ModelStatus.HEURISTIC_ONLY: (
        "No checkpoint is available. The score comes from a fixed "
        "signal-processing baseline, not from the CNN. It is a comparison "
        "baseline only."
    ),
}


@dataclass
class ScoringResult:
    spoof_probability: float
    scorer: str
    decision_threshold: float
    windows_scored: int
    per_window_probabilities: list
    evidence: Dict[str, Any]
    caveats: list


class AcousticBaseline:
    """
    Deterministic signal-processing baseline.

    This exists to answer the question a reviewer will ask - "does the CNN beat
    a simple hand-written rule?" - and to give the service something honest to
    return when no checkpoint exists. It is not a fallback that impersonates
    the model; results from it are labelled with scorer="acoustic_baseline".

    The features are the ones the audio-forensics literature reports as
    discriminative: vocoders tend to under-populate the top of the band, and
    their formant tracks vary less over time than a human vocal tract does.
    """

    @staticmethod
    def features(spectrogram: np.ndarray, waveform: np.ndarray) -> Dict[str, float]:
        upper = float(np.mean(spectrogram[96:, :]))
        lower = float(np.mean(spectrogram[:32, :]))
        energy_ratio = upper / (lower + 1e-6)
        temporal_variance = float(np.var(spectrogram, axis=1).mean())
        band_variance = float(np.var(spectrogram, axis=0).mean())
        zcr = float(np.mean(np.abs(np.diff(np.sign(waveform)))) / 2.0)
        flatness = float(
            np.exp(np.mean(np.log(spectrogram + 1e-8))) / (np.mean(spectrogram) + 1e-8)
        )
        return {
            "upper_band_energy": round(upper, 4),
            "lower_band_energy": round(lower, 4),
            "high_low_energy_ratio": round(energy_ratio, 4),
            "temporal_variance": round(temporal_variance, 4),
            "band_variance": round(band_variance, 4),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_flatness": round(flatness, 4),
        }

    @staticmethod
    def score(feats: Dict[str, float]) -> float:
        rigidity = 1.0 - min(1.0, feats["temporal_variance"] * 25.0)
        band_anomaly = min(1.0, feats["high_low_energy_ratio"] * 3.0)
        return float(np.clip(0.5 * rigidity + 0.5 * band_anomaly, 0.02, 0.98))


class ModelRuntime:
    """Loads the checkpoint once and scores clips window by window."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VoiceDeepfakeCNN(dropout_rate=config.DROPOUT_RATE).to(self.device)
        self.model.eval()

        self.weights_path = Path(weights_path or config.WEIGHTS_PATH)
        self.meta_path = Path(meta_path or config.META_PATH)

        self.meta: Dict[str, Any] = {}
        self.status: str = ModelStatus.HEURISTIC_ONLY
        self.decision_threshold: float = config.DEFAULT_THRESHOLD
        self.weights_loaded = False

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not read model metadata: %s", exc)
                self.meta = {}

        if not self.weights_path.exists():
            log.warning(
                "no checkpoint at %s - serving the acoustic baseline",
                self.weights_path,
            )
            self.status = ModelStatus.HEURISTIC_ONLY
            return

        try:
            # weights_only=True refuses to execute arbitrary pickled objects.
            # A .pth is an untrusted input if it did not come from this repo.
            state = torch.load(
                self.weights_path, map_location=self.device, weights_only=True
            )
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.model.eval()
            self.weights_loaded = True
        except Exception as exc:                       # noqa: BLE001
            log.error("checkpoint at %s failed to load: %s", self.weights_path, exc)
            self.status = ModelStatus.HEURISTIC_ONLY
            return

        # Provenance decides trust. A checkpoint with no metadata sidecar has
        # no verifiable training history, so it is treated as a demo artefact
        # rather than assumed to be trained on a real corpus.
        declared = str(self.meta.get("training_data_kind", "")).lower()
        if declared == "real_corpus":
            self.status = ModelStatus.TRAINED
        elif declared in {"synthetic_demo", "generated_tones"}:
            self.status = ModelStatus.DEMO_SYNTHETIC
        else:
            log.warning(
                "checkpoint has no training provenance metadata (%s); "
                "treating it as a demo artefact",
                self.meta_path,
            )
            self.status = ModelStatus.DEMO_SYNTHETIC

        self.decision_threshold = float(
            self.meta.get("decision_threshold", config.DEFAULT_THRESHOLD)
        )

    # ------------------------------------------------------------------
    @property
    def caveats(self) -> list:
        text = CAVEAT_TEXT.get(self.status)
        return [text] if text else []

    def describe(self) -> Dict[str, Any]:
        """Provenance block; surfaced by /api/health and every prediction."""
        return {
            "status": self.status,
            "trustworthy": self.status == ModelStatus.TRAINED,
            "weights_loaded": self.weights_loaded,
            "device": str(self.device),
            "parameters": count_parameters(self.model),
            "decision_threshold": round(self.decision_threshold, 4),
            "trained_on": self.meta.get("dataset_name", "unknown"),
            "training_data_kind": self.meta.get("training_data_kind", "unknown"),
            "trained_at": self.meta.get("trained_at", "unknown"),
            "reported_metrics": self.meta.get("metrics", {}),
            "caveats": self.caveats,
        }

    # ------------------------------------------------------------------
    def score_clip(
        self,
        batch: np.ndarray,
        preview_spectrogram: np.ndarray,
        waveform: np.ndarray,
    ) -> ScoringResult:
        """
        Score every analysis window and aggregate to one clip-level number.

        Aggregation uses the 90th percentile of the per-window scores rather
        than the mean or the max. The mean lets a long stretch of genuine
        speech bury a short injected clone; the max fires on a single noisy
        window. A high percentile keeps sensitivity to a localised attack while
        still requiring more than one anomalous frame.
        """
        feats = AcousticBaseline.features(preview_spectrogram, waveform)

        if self.weights_loaded:
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            with torch.no_grad():
                probs = self.model.predict_proba(tensor).cpu().numpy().ravel()
            probs = probs.astype(float)
            clip_prob = float(np.percentile(probs, 90)) if probs.size > 1 else float(probs[0])
            scorer = "cnn"
        else:
            probs = np.array([AcousticBaseline.score(feats)], dtype=float)
            clip_prob = float(probs[0])
            scorer = "acoustic_baseline"

        return ScoringResult(
            spoof_probability=clip_prob,
            scorer=scorer,
            decision_threshold=self.decision_threshold,
            windows_scored=int(batch.shape[0]),
            per_window_probabilities=[round(float(p), 4) for p in probs[:64]],
            evidence=feats,
            caveats=self.caveats,
        )


# Module-level singleton: the checkpoint is loaded once per process, not per
# request. Import this, do not construct ModelRuntime in a handler.
runtime = ModelRuntime()
