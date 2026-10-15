"""
VoxShield REST API.

Endpoints
---------
GET  /api/health              service and model provenance
GET  /api/profiles            deployment profiles the caller may select
GET  /api/samples             demo clips bundled with the repository
GET  /api/samples/{id}/audio  stream a demo clip for playback
POST /api/samples/{id}/analyze  score a demo clip
POST /api/analyze             score an uploaded clip
POST /api/feedback            queue an operator's correction (opt-in, consent required)
GET  /api/feedback/summary    size of the correction queue

The service is stateless. Apart from corrections an operator explicitly
submits with consent (backend/feedback.py), it stores nothing: uploads are written to a
short-lived temporary file only because the audio decoders want a seekable
path, and that file is removed in a finally block. Nothing about the caller or
the audio is logged beyond a size and a duration, which matters because the
input is by definition someone's voice.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Allow `python -m uvicorn backend.app:app` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from backend import feedback, security
from backend.model_loader import runtime
from backend.preprocessor import AudioPreprocessor
from backend.risk_engine import RiskScoringEngine, list_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("voxshield")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLES_DIR = BASE_DIR / "test_samples"

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log model provenance once, loudly, at start-up."""
    described = runtime.describe()
    log.info("model status: %s (trained on: %s)",
             described["status"], described["trained_on"])
    for caveat in described["caveats"]:
        log.warning("MODEL CAVEAT: %s", caveat)
    yield


app = FastAPI(
    lifespan=lifespan,
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=(
        "Spectrogram-CNN detection of synthetic and cloned speech, exposed as "
        "a risk-scoring service for any organisation that accepts voice input."
    ),
)

# An explicit allowlist. "*" with allow_credentials=True is rejected by
# browsers and is an anti-pattern regardless; when no origins are configured
# the API is same-origin only, which is how the bundled dashboard runs.
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

preprocessor = AudioPreprocessor()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Baseline hardening headers on every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


# ----------------------------------------------------------------------
# Informational endpoints
# ----------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "model": runtime.describe(),
        "audio_frontend": {
            "sample_rate_hz": preprocessor.sample_rate,
            "window_seconds": preprocessor.window_seconds,
            "window_overlap": config.WINDOW_OVERLAP,
            "spectrogram": f"{preprocessor.n_mels}x{preprocessor.target_frames}",
            "n_fft": preprocessor.n_fft,
            "hop_length": preprocessor.hop_length,
        },
        "limits": {
            "max_upload_bytes": config.MAX_UPLOAD_BYTES,
            "max_audio_seconds": config.MAX_DECODED_SECONDS,
            "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
        },
    }


@app.get("/api/profiles")
async def profiles() -> dict:
    return {"profiles": list_profiles()}


@app.get("/api/samples")
async def samples() -> dict:
    found = []
    if SAMPLES_DIR.exists():
        for path in sorted(SAMPLES_DIR.glob("*.wav")):
            found.append(
                {
                    "id": path.name,
                    "name": path.stem.replace("_", " ").title(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "samples": found,
        "note": (
            "Real recordings from the held-out test sets: two genuine voices "
            "(ASVspoof 2019 and LibriSpeech) and two clones (attacks A07 and A16)."
        ),
    }


@app.get("/api/samples/{sample_id}/audio")
async def sample_audio(sample_id: str):
    """
    Stream a demo clip so the dashboard can play it.

    v1 pointed the <audio> element at the prediction endpoint, which is
    POST-only, so playback always failed with 405. Serving the media from its
    own GET route is both correct and cacheable.
    """
    try:
        path = security.resolve_sample(sample_id, SAMPLES_DIR)
    except security.UploadRejected as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav")


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def _analyze_path(path: str, filename: str, profile: Optional[str]) -> dict:
    started = time.perf_counter()

    waveform, batch, preview = preprocessor.process(path)
    scoring = runtime.score_clip(batch, preview, waveform)
    assessment = RiskScoringEngine.assess(
        spoof_probability=scoring.spoof_probability,
        decision_threshold=scoring.decision_threshold,
        profile_key=profile,
        evidence=scoring.evidence,
        caveats=scoring.caveats,
    )

    return {
        "filename": filename,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "audio": {
            "duration_seconds": round(len(waveform) / preprocessor.sample_rate, 2),
            "windows_scored": scoring.windows_scored,
            "per_window_probabilities": scoring.per_window_probabilities,
        },
        "scorer": scoring.scorer,
        "model": runtime.describe(),
        "risk_assessment": assessment.to_dict(),
        "spectrogram_image": preprocessor.spectrogram_png_base64(preview),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    profile: str = Form("generic"),
) -> dict:
    """Score an uploaded recording."""
    temp_path = None
    try:
        suffix = security.validate_extension(file.filename)
        content = await file.read(config.MAX_UPLOAD_BYTES + 1)
        security.validate_payload(content, suffix)
    except security.UploadRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with tempfile.NamedTemporaryFile(
            suffix=security.safe_temp_suffix(suffix), delete=False
        ) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        # The original filename is echoed back to the caller but never used to
        # build a path, and it is not written to the log.
        return _analyze_path(temp_path, file.filename or "upload", profile)

    except ValueError as exc:
        # Raised by the preprocessor for undecodable or over-long audio. The
        # message is ours, so it is safe to return.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:                            # noqa: BLE001
        # Anything else is an internal fault. The detail is logged with a
        # correlation id and a generic message is returned, because decoder
        # exceptions carry absolute filesystem paths.
        incident = uuid.uuid4().hex[:12]
        log.exception("analysis failed [incident=%s]", incident)
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed. Reference {incident}.",
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                log.warning("could not remove temporary file")


@app.post("/api/feedback")
async def submit_feedback(
    file: UploadFile = File(...),
    label: str = Form(...),
    consent: bool = Form(False),
    model_score: Optional[float] = Form(None),
    x_feedback_token: Optional[str] = Header(None),
) -> dict:
    """
    Queue an operator's correction: the clip and its true label.

    Nothing is learned here. The clip waits in a review queue until
    fine_tune_feedback.py is run, and that script only installs a new model
    if it is no worse on held-out data.
    """
    content = await file.read(config.MAX_UPLOAD_BYTES + 1)
    try:
        fid = feedback.save(content, file.filename, label, consent,
                            model_score=model_score, token=x_feedback_token)
    except feedback.FeedbackRejected as exc:
        status = 403 if "token" in str(exc) or "disabled" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    log.info("feedback queued (label=%s)", label)
    return {"queued": True, "id": fid, "pending": feedback.pending_count()}


@app.get("/api/feedback/summary")
async def feedback_summary() -> dict:
    return {
        "enabled": config.FEEDBACK_ENABLED,
        "token_required": bool(config.FEEDBACK_TOKEN),
        "pending": feedback.pending_count(),
        "max_pending": config.FEEDBACK_MAX_PENDING,
    }


@app.post("/api/samples/{sample_id}/analyze")
async def analyze_sample(sample_id: str, profile: str = Form("generic")) -> dict:
    """Score one of the bundled demo clips."""
    try:
        path = security.resolve_sample(sample_id, SAMPLES_DIR)
    except security.UploadRejected as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        return _analyze_path(str(path), sample_id, profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:                            # noqa: BLE001
        incident = uuid.uuid4().hex[:12]
        log.exception("sample analysis failed [incident=%s]", incident)
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed. Reference {incident}.",
        ) from exc


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
