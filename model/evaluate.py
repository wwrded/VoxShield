"""
Detection metrics.

Accuracy is the wrong headline number for this problem and the report should
not lead with it. ASVspoof LA is about 90% spoof by construction, so a model
that answers "spoof" unconditionally scores 90% accuracy while detecting
nothing. Two metrics carry the real information:

  Equal Error Rate (EER)
      The operating point where the false-acceptance rate (a clone accepted as
      genuine) equals the false-rejection rate (a real person refused). It is
      threshold-free, which is what makes it comparable across published
      systems, and it is the number the ASVspoof challenge series reports.

  ROC-AUC
      The probability that a randomly chosen spoof scores above a randomly
      chosen bonafide utterance. Also threshold-free, and insensitive to the
      class imbalance.

The official ASVspoof metric is minimum t-DCF, which weights detection errors
by their cost inside a full speaker-verification system. Computing it requires
the ASV subsystem scores distributed with the challenge; this module reports
EER and AUC, which is the standard practice for countermeasure-only work, and
the report should say so rather than implying t-DCF was computed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Equal Error Rate and the threshold that achieves it.

    The crossing of FPR and FNR rarely lands exactly on an ROC vertex, so the
    rate is linearly interpolated between the two bracketing points rather
    than snapped to the nearest one. Snapping - what v1 did with
    `argmin(|fnr - fpr|)` alone - biases the reported EER low on small
    evaluation sets, which is exactly the direction that flatters a result.

    scikit-learn prepends an infinite threshold to `thresholds` so that the
    first ROC point is "predict nothing positive". That entry is dropped here;
    returning it as a decision threshold would make the model classify
    everything as bonafide.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    if np.unique(y_true).size < 2:
        raise ValueError("EER needs both bonafide and spoof examples")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    fnr = 1.0 - tpr

    diff = fnr - fpr
    idx = int(np.nanargmin(np.abs(diff)))

    # Interpolate between idx and its neighbour on the other side of the crossing.
    neighbour = idx + 1 if idx + 1 < len(diff) and diff[idx] * diff[idx + 1] < 0 else idx - 1
    if 0 <= neighbour < len(diff) and diff[idx] != diff[neighbour]:
        alpha = diff[idx] / (diff[idx] - diff[neighbour])
        eer = float(fpr[idx] + alpha * (fpr[neighbour] - fpr[idx]))
        threshold = float(thresholds[idx] + alpha * (thresholds[neighbour] - thresholds[idx]))
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        threshold = float(thresholds[idx])

    return max(0.0, min(1.0, eer)), threshold


def evaluate_scores(
    y_true: Sequence[float],
    y_scores: Sequence[float],
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """
    Full metric suite.

    If `threshold` is None the EER operating point is used, which is the
    threshold the deployed service should also use. Reporting accuracy at 0.5
    while quoting EER from a different point is a common and misleading
    mismatch.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    eer, eer_threshold = compute_eer(y_true, y_scores)
    used_threshold = float(eer_threshold if threshold is None else threshold)
    y_pred = (y_scores >= used_threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    return {
        "eer": float(eer),
        "eer_percent": round(float(eer) * 100, 3),
        "eer_threshold": float(eer_threshold),
        "decision_threshold": used_threshold,
        "roc_auc": float(auc(fpr, tpr)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_spoof": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, y_pred, zero_division=0)),
        # Positive = spoof, so fn is a clone that was accepted as genuine (the
        # security failure) and fp is a genuine speaker who was refused (the
        # usability failure). Naming these the other way round - which is easy
        # to do, because "positive" and "accepted" pull in opposite directions
        # here - reverses the meaning of the two numbers a reader most needs to
        # weigh against each other.
        "false_acceptance_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_rejection_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "counts": {
            "bonafide": int((y_true == 0).sum()),
            "spoof": int((y_true == 1).sum()),
        },
        "confusion_matrix": {
            "true_negative_bonafide_passed": tn,
            "false_positive_genuine_rejected": fp,
            "false_negative_clone_accepted": fn,
            "true_positive_clone_detected": tp,
        },
    }


def per_attack_eer(
    y_true: Sequence[float],
    y_scores: Sequence[float],
    attack_ids: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    EER for each spoofing system, scored against the shared bonafide pool.

    This is the breakdown reviewers will ask for, because a single headline
    EER hides the interesting result: a countermeasure typically handles the
    attacks that resemble its training data and collapses on one or two
    unfamiliar vocoders. ASVspoof 2019 LA is designed for exactly this
    analysis - the eval partition contains attack types A07-A19 that are
    absent from the training partition A01-A06.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_scores = np.asarray(y_scores, dtype=float).ravel()
    attack_ids = np.asarray(attack_ids)

    bona_mask = y_true == 0
    bona_scores = y_scores[bona_mask]

    results: Dict[str, Dict[str, float]] = {}
    for attack in sorted({a for a in attack_ids[y_true == 1] if a not in ("-", "")}):
        mask = (attack_ids == attack) & (y_true == 1)
        if not mask.any():
            continue
        scores = np.concatenate([bona_scores, y_scores[mask]])
        labels = np.concatenate([np.zeros(bona_scores.size), np.ones(int(mask.sum()))])
        try:
            eer, threshold = compute_eer(labels, scores)
        except ValueError:
            continue
        results[attack] = {
            "eer_percent": round(eer * 100, 3),
            "threshold": round(threshold, 4),
            "n_spoof": int(mask.sum()),
        }
    return results


def save_roc_curve(
    y_true: Sequence[float],
    y_scores: Sequence[float],
    out_path: str,
    title: str = "Synthetic speech detection - ROC",
) -> str:
    """Write a publication-ready ROC figure. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    y_true = np.asarray(y_true).ravel().astype(int)
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    eer, _ = compute_eer(y_true, y_scores)

    fig = Figure(figsize=(6, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, color="#1e3a8a", lw=2, label=f"CNN (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=1, label="Chance")
    ax.scatter([eer], [1 - eer], color="#ef4444", zorder=5,
               label=f"EER = {eer * 100:.2f}%")
    # With positive = spoof, the x axis is bonafide utterances flagged as
    # spoof, i.e. genuine speakers refused - a false rejection, not a false
    # acceptance.
    ax.set_xlabel("False rejection rate (genuine speaker refused)")
    ax.set_ylabel("True positive rate (clone detected)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


def format_report(metrics: Dict[str, object], per_attack: Optional[Dict] = None) -> str:
    """Human-readable metric block for the logbook and the report."""
    lines: List[str] = [
        "=" * 62,
        f"  bonafide utterances : {metrics['counts']['bonafide']}",
        f"  spoof utterances    : {metrics['counts']['spoof']}",
        "-" * 62,
        f"  EER                 : {metrics['eer_percent']:.3f} %"
        f"  (threshold {metrics['eer_threshold']:.4f})",
        f"  ROC-AUC             : {metrics['roc_auc']:.4f}",
        f"  Balanced accuracy   : {metrics['balanced_accuracy'] * 100:.2f} %",
        f"  Accuracy            : {metrics['accuracy'] * 100:.2f} %"
        "   <- imbalance-sensitive, do not headline",
        f"  Spoof recall        : {metrics['recall_spoof'] * 100:.2f} %",
        f"  False acceptance    : {metrics['false_acceptance_rate'] * 100:.2f} %"
        "   (clones accepted as genuine)",
        f"  False rejection     : {metrics['false_rejection_rate'] * 100:.2f} %"
        "   (genuine speakers refused)",
        "=" * 62,
    ]
    if per_attack:
        lines.append("  Per-attack EER")
        for attack, values in per_attack.items():
            lines.append(
                f"    {attack:<5} {values['eer_percent']:>7.3f} %"
                f"   (n = {values['n_spoof']})"
            )
        lines.append("=" * 62)
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    truth = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.beta(1, 5, 50), rng.beta(5, 1, 50)])
    print(format_report(evaluate_scores(truth, scores)))
