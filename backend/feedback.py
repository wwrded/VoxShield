"""
Operator feedback: corrections that can later fine-tune the model.

When an operator knows the true answer for a clip (a confirmed genuine
customer, a confirmed fraud attempt), they can send it back with the correct
label. The clip and its label wait in a review queue on disk; nothing is
learned until someone runs fine_tune_feedback.py, which only installs the
updated model if it is no worse on held-out data.

This is the one place the service keeps audio, so the rules are strict:

* the deployment must enable it (config.FEEDBACK_ENABLED);
* the caller must confirm consent for this clip;
* when a token is configured, the caller must present it;
* the queue is capped, so it cannot be used to fill the disk;
* only a random id, the label, the model's score and a timestamp are stored,
  never the original filename or anything about the caller.
"""

from __future__ import annotations

import hmac
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from backend import security

LABELS = {"genuine": 0, "synthetic": 1}


class FeedbackRejected(ValueError):
    """The correction was refused; the message is safe to return."""


def pending_dir(root: Optional[Path] = None) -> Path:
    return Path(root or config.FEEDBACK_DIR) / "pending"


def pending_count(root: Optional[Path] = None) -> int:
    d = pending_dir(root)
    return len(list(d.glob("*.json"))) if d.exists() else 0


def check_token(supplied: Optional[str]) -> None:
    expected = config.FEEDBACK_TOKEN
    if expected and not hmac.compare_digest(supplied or "", expected):
        raise FeedbackRejected("a valid feedback token is required")


def save(content: bytes, filename: Optional[str], label: str, consent: bool,
         model_score: Optional[float] = None, token: Optional[str] = None,
         root: Optional[Path] = None) -> str:
    """Validate a correction and queue it. Returns the feedback id."""
    if not config.FEEDBACK_ENABLED:
        raise FeedbackRejected("feedback is disabled on this deployment")
    check_token(token)
    if not consent:
        raise FeedbackRejected("consent to keep this recording is required")
    if label not in LABELS:
        raise FeedbackRejected("label must be 'genuine' or 'synthetic'")
    if pending_count(root) >= config.FEEDBACK_MAX_PENDING:
        raise FeedbackRejected("the review queue is full; fine-tune or clear it first")

    try:
        suffix = security.validate_extension(filename)
        security.validate_payload(content, suffix)
    except security.UploadRejected as exc:
        raise FeedbackRejected(str(exc)) from exc

    if model_score is not None and not 0.0 <= float(model_score) <= 1.0:
        model_score = None

    d = pending_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    fid = uuid.uuid4().hex
    (d / f"{fid}{suffix}").write_bytes(content)
    (d / f"{fid}.json").write_text(json.dumps({
        "id": fid,
        "label": label,
        "audio": f"{fid}{suffix}",
        "model_score": None if model_score is None else round(float(model_score), 4),
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    return fid


def load_pending(root: Optional[Path] = None) -> list:
    """All queued corrections as dicts with an absolute 'path'."""
    d = pending_dir(root)
    items = []
    if d.exists():
        for meta in sorted(d.glob("*.json")):
            try:
                rec = json.loads(meta.read_text(encoding="utf-8"))
                audio = d / Path(rec["audio"]).name     # never follow a path out of the queue
                if audio.exists() and rec.get("label") in LABELS:
                    rec["path"] = str(audio)
                    items.append(rec)
            except (OSError, ValueError, KeyError):
                continue
    return items
