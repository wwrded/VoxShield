# VoxShield — Synthetic Voice Fraud Detection

Detection of AI-generated, cloned and voice-converted speech, exposed as a
risk-scoring API that any organisation accepting voice as an input channel can
call.

**Information Security Project · BSc (Hons) Information Technology
(Cyber Security) · Sri Lanka Institute of Information Technology**

| Team member |
|---|
| E.M.A.D. Ekanayake |
| H.M.A.A. Herath |
| W.W. Rathnayake |

Supervisor: Dr. Janaka Alawatugoda

---

## Project status and limitations

VoxShield is a **research prototype**, not a production fraud-detection product.

The checkpoint in `model/saved_models/` is trained on a real corpus (see
`model/saved_models/model_meta.json`, `training_data_kind: "real_corpus"`):
ASVspoof 2019 LA training data (attacks A01-A06) plus LibriSpeech real and
vocoded speech, 34,204 training clips in total.

Measured on held-out ASVspoof 2019 LA attacks A07-A19, which never appear in
training:

| Metric | Value |
|---|---|
| Equal Error Rate (EER) | 21.9% |
| ROC-AUC | 0.857 |
| Genuine speech wrongly refused (deployed threshold) | 2.1% |
| Clones accepted (deployed threshold) | 45.6% |

Performance also drops under telephone-style conditions
(`docs/robustness_report.json`): about 31.7% EER with a telephone filter, 30.3%
with G.711 and 34.2% with 20 dB noise. Results differ sharply by attack type
(per-attack EER is in the same report), and Sinhala and Tamil speech has not
been measured. Do not use VoxShield as the sole basis for any decision about a
person. See `docs/MODEL_CARD.md` for intended use and known limitations.

If you retrain, copy both `voice_deepfake_cnn.pth` **and** `model_meta.json`
into `model/saved_models/`. Only a checkpoint whose metadata declares
`training_data_kind: "real_corpus"` is treated as trained, and `/api/health`
reports the provenance of whatever is loaded.

---

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>. On Windows, `run_app.bat` does both steps.

Run the tests:

```bash
python -m pytest -q
```

---

## Architecture

```
 audio in ──▶ AudioPreprocessor ──▶ VoiceDeepfakeCNN ──▶ RiskScoringEngine ──▶ JSON
             16 kHz mono            4 conv blocks        deployment profile
             3 s windows, 50%       913 k parameters     calibrated threshold
             overlap                one logit/window     tier + action + CIA
             128×128 log-Mel        p90 aggregation
```

| Path | Responsibility |
|---|---|
| `config.py` | Every tunable constant. Imported by training *and* serving so the two cannot drift. |
| `backend/preprocessor.py` | Decode, peak-normalise, window, log-Mel, render. |
| `backend/model_loader.py` | Checkpoint loading, provenance, windowed scoring, acoustic baseline. |
| `backend/risk_engine.py` | Probability → tier, action and CIA impact, per deployment profile. |
| `backend/security.py` | Upload validation, magic-byte sniffing, path-traversal defence. |
| `backend/app.py` | FastAPI routes, error handling, security headers. |
| `model/cnn_architecture.py` | The network. |
| `model/datasets.py` | ASVspoof protocol reader, folder corpus, labelled demo generator. |
| `model/train.py` | Runnable training CLI; writes weights **and** provenance metadata. |
| `model/evaluate.py` | EER (interpolated), ROC-AUC, per-attack breakdown, ROC figure. |
| `frontend/` | Dashboard: profile selector, per-window chart, provenance banner. |
| `tests/` | 73 tests covering the front-end, metrics, risk mapping, API and defences. |

---

## Training

```bash
# The real benchmark — this is what the report should quote
python -m model.train --dataset asvspoof --data-root ./LA --epochs 20

# A locally collected corpus laid out as bonafide/ and spoof/
python -m model.train --dataset folder --data-root ./my_corpus

# Pipeline smoke test on generated tones — NOT a detector
python -m model.train --dataset synthetic-demo --epochs 3
```

Every run writes `model_meta.json` beside the weights recording the corpus, the
Equal Error Rate threshold and the metrics actually measured.

---

## Deployment profiles

The same detector serves different processes with different risk appetites, so
the operating point is a parameter rather than a constant.

| Profile | Escalates at | Rationale |
|---|---|---|
| `generic` | 1.00 × threshold | Balanced default. |
| `contact_center` | 1.10 × threshold | High volume; false alarms cost handling time. |
| `financial_services` | 0.75 × threshold | Loss per missed clone is large. |
| `enterprise_helpdesk` | 0.85 × threshold | Protects credential and MFA resets. |
| `kyc_onboarding` | 0.70 × threshold | A missed clone creates a durable fraudulent identity. |
| `media_verification` | 0.90 × threshold | Reports evidence; never blocks. |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service state and model provenance |
| `GET` | `/api/profiles` | Deployment profile catalogue |
| `GET` | `/api/samples` | Bundled demo clips |
| `GET` | `/api/samples/{id}/audio` | Stream a demo clip |
| `POST` | `/api/samples/{id}/analyze` | Score a demo clip |
| `POST` | `/api/analyze` | Score an upload (`file`, `profile`) |

```bash
curl -F "file=@call.wav" -F "profile=financial_services" \
     http://127.0.0.1:8000/api/analyze
```

Every response names the scorer that produced the number (`cnn` or
`acoustic_baseline`) and carries the model's provenance block.

---

## Security posture

This service accepts attacker-controlled binary input and hands it to a media
decoder, so the input path is treated as hostile:

- 25 MB upload cap, enforced during the read
- extension allowlist plus magic-byte verification
- sanitised temporary filenames; nothing user-controlled reaches a path
- path-traversal defence on the sample endpoints, with tests
- internal errors return a correlation id, never a filesystem path
- `weights_only=True` on checkpoint load, so a hostile `.pth` cannot execute code
- explicit CORS allowlist; never `*` with credentials
- `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Cache-Control`
- stateless: audio is deleted in a `finally` block and never logged

---

## Documentation

- `docs/MODEL_CARD.md` — intended use, limits, failure modes
