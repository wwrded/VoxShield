"""
Central configuration for the VoxShield synthetic-voice detection system.

Every tunable constant lives here so that the training pipeline, the inference
runtime and the REST API cannot drift apart. Training/serving skew (training on
one feature representation and serving on another) is the single most common
cause of a model that scores well offline and fails in production, so the
audio front-end parameters below are imported by both sides rather than being
re-declared in each place.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Audio front-end (shared by training and inference - do not fork these)
# --------------------------------------------------------------------------
SAMPLE_RATE = 16_000        # Hz. ASVspoof LA/DF is distributed at 16 kHz.
WINDOW_SECONDS = 3.0        # Length of one analysis window.
N_MELS = 128                # Mel filterbank size -> spectrogram height.
N_FFT = 1024                # STFT window (64 ms at 16 kHz).
HOP_LENGTH = 375            # Chosen so 3.0 s yields ~128 frames.
TARGET_FRAMES = 128         # Spectrogram width after pad/trim.
FMIN = 20                   # Hz. Below this is DC rumble, not speech.
FMAX = SAMPLE_RATE // 2     # Nyquist. Vocoder artefacts live in the top band.

# Sliding-window inference: real recordings are far longer than one window.
WINDOW_OVERLAP = 0.5        # 50% overlap between consecutive windows.
MAX_WINDOWS_PER_CLIP = 40   # Bounds worst-case CPU time on a long upload.

# Silence handling (shared by training and inference).
TRIM_TOP_DB = 35            # Frames 35 dB below the loudest frame count as silence.
MIN_SPEECH_SECONDS = 0.5    # Never trim a clip below this length.
SILENT_WINDOW_RATIO = 0.10  # Drop windows quieter than 10% of the loudest window.

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
DROPOUT_RATE = 0.25
WEIGHTS_PATH = BASE_DIR / "model" / "saved_models" / "voice_deepfake_cnn.pth"
META_PATH = BASE_DIR / "model" / "saved_models" / "model_meta.json"

# Decision threshold used when a trained checkpoint carries no calibrated
# threshold of its own. 0.5 is the naive default; training overwrites it with
# the Equal Error Rate operating point.
DEFAULT_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Upload limits (defensive: this service accepts untrusted binary input)
# --------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("VOXSHIELD_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
MAX_DECODED_SECONDS = 300.0   # Reject absurdly long audio before decoding cost.

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
# CORS: an explicit allowlist, never "*" combined with credentials. Override in
# deployment with VOXSHIELD_CORS_ORIGINS="https://fraud-ops.example.com,..."
_cors_env = os.getenv("VOXSHIELD_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]

# --------------------------------------------------------------------------
# Feedback (operator corrections used for fine-tuning)
# --------------------------------------------------------------------------
# Feedback is the only path that keeps audio, so it is opt-in three times
# over: the deployment enables it, the operator ticks a consent box, and the
# clip waits in a review queue until someone fine-tunes with it. When a token
# is configured, only callers that send it may submit corrections, which
# limits who can try to poison the training data.
FEEDBACK_ENABLED = os.getenv("VOXSHIELD_FEEDBACK", "1") == "1"
FEEDBACK_TOKEN = os.getenv("VOXSHIELD_FEEDBACK_TOKEN", "")
FEEDBACK_DIR = Path(os.getenv("VOXSHIELD_FEEDBACK_DIR", str(BASE_DIR / "feedback")))
FEEDBACK_MAX_PENDING = 500

API_TITLE = "VoxShield Synthetic Voice Detection API"
API_VERSION = "2.0.0"
