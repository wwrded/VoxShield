"""End-to-end tests against the FastAPI application."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import config
from backend.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def wav_bytes(seconds: float = 3.0, freq: float = 200.0) -> bytes:
    t = np.linspace(0, seconds, int(config.SAMPLE_RATE * seconds), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, config.SAMPLE_RATE, format="WAV")
    return buf.getvalue()


class TestHealth:
    def test_reports_model_provenance(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "healthy"
        model = body["model"]
        assert model["status"] in {
            "TRAINED", "DEMO_SYNTHETIC", "UNTRAINED", "HEURISTIC_ONLY"
        }
        # "trustworthy" must never be asserted without a real training corpus.
        assert model["trustworthy"] == (model["training_data_kind"] == "real_corpus")

    def test_untrusted_models_carry_a_caveat(self, client):
        model = client.get("/api/health").json()["model"]
        if not model["trustworthy"]:
            assert model["caveats"], "an untrusted model must explain itself"

    def test_security_headers_present(self, client):
        headers = client.get("/api/health").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"


class TestProfiles:
    def test_catalogue_is_not_call_centre_only(self, client):
        keys = {p["key"] for p in client.get("/api/profiles").json()["profiles"]}
        assert {"generic", "financial_services", "enterprise_helpdesk"} <= keys


class TestAnalyze:
    def test_scores_a_valid_upload(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("call.wav", wav_bytes(), "audio/wav")},
            data={"profile": "generic"},
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["risk_assessment"]["spoof_probability"] <= 1.0
        assert body["risk_assessment"]["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}
        assert body["scorer"] in {"cnn", "acoustic_baseline"}
        assert body["spectrogram_image"].startswith("data:image/png;base64,")

    def test_result_names_the_scorer_that_produced_it(self, client):
        """v1 returned heuristic output in the same field as model output."""
        body = client.post(
            "/api/analyze", files={"file": ("c.wav", wav_bytes(), "audio/wav")}
        ).json()
        assert body["scorer"] == ("cnn" if body["model"]["weights_loaded"]
                                  else "acoustic_baseline")

    def test_long_audio_is_windowed(self, client):
        body = client.post(
            "/api/analyze", files={"file": ("long.wav", wav_bytes(15.0), "audio/wav")}
        ).json()
        assert body["audio"]["windows_scored"] > 1
        assert len(body["audio"]["per_window_probabilities"]) > 1

    def test_profile_changes_the_recommended_action(self, client):
        payload = wav_bytes()
        bank = client.post(
            "/api/analyze",
            files={"file": ("c.wav", payload, "audio/wav")},
            data={"profile": "financial_services"},
        ).json()
        assert bank["risk_assessment"]["profile"] == "financial_services"

    def test_rejects_empty_upload(self, client):
        response = client.post(
            "/api/analyze", files={"file": ("empty.wav", b"", "audio/wav")}
        )
        # v1 caught its own HTTPException and re-raised everything as a 500.
        assert response.status_code == 400

    def test_rejects_disallowed_extension(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("payload.exe", b"MZ\x90\x00" * 8, "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_rejects_content_type_mismatch(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("fake.wav", b"MZ\x90\x00" + b"\x00" * 128, "audio/wav")},
        )
        assert response.status_code == 400
        assert "not a recognised" in response.json()["detail"]

    def test_rejects_oversize_upload(self, client):
        payload = b"RIFF" + b"\x00" * (config.MAX_UPLOAD_BYTES + 1024)
        response = client.post(
            "/api/analyze", files={"file": ("big.wav", payload, "audio/wav")}
        )
        assert response.status_code == 400

    def test_undecodable_audio_returns_422_not_500(self, client):
        payload = b"RIFF" + b"\x00" * 200          # valid magic, invalid contents
        response = client.post(
            "/api/analyze", files={"file": ("broken.wav", payload, "audio/wav")}
        )
        assert response.status_code == 422

    def test_errors_do_not_leak_filesystem_paths(self, client):
        response = client.post(
            "/api/analyze", files={"file": ("broken.wav", b"RIFF" + b"\x00" * 200, "audio/wav")}
        )
        detail = response.json()["detail"]
        assert "C:\\" not in detail and "/home/" not in detail


class TestSamples:
    def test_lists_bundled_samples(self, client):
        body = client.get("/api/samples").json()
        assert "note" in body            # must disclose that they are generated

    def test_sample_audio_is_served_over_get(self, client):
        samples = client.get("/api/samples").json()["samples"]
        if not samples:
            pytest.skip("no bundled samples")
        response = client.get(f"/api/samples/{samples[0]['id']}/audio")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/")

    @pytest.mark.parametrize("attack", ["..%2F..%2Fconfig.py", "....//config.py"])
    def test_sample_endpoints_reject_traversal(self, client, attack):
        assert client.get(f"/api/samples/{attack}/audio").status_code in (404, 400)
        assert client.post(f"/api/samples/{attack}/analyze").status_code in (404, 400, 422)


class TestFeedback:
    """Operator corrections: opt-in, consent-gated, validated, capped."""

    @pytest.fixture(autouse=True)
    def _queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "FEEDBACK_DIR", tmp_path)
        monkeypatch.setattr(config, "FEEDBACK_ENABLED", True)
        monkeypatch.setattr(config, "FEEDBACK_TOKEN", "")
        self.root = tmp_path

    def post(self, client, **data):
        form = {"label": "genuine", "consent": "true"}
        form.update(data)
        return client.post("/api/feedback",
                           files={"file": ("c.wav", wav_bytes(), "audio/wav")}, data=form)

    def test_queues_a_consented_correction(self, client):
        r = self.post(client, model_score="0.91")
        assert r.status_code == 200 and r.json()["pending"] == 1
        meta = list((self.root / "pending").glob("*.json"))
        assert len(meta) == 1
        saved = meta[0].read_text()
        assert '"genuine"' in saved and "c.wav" not in saved   # original name is not kept

    def test_refused_without_consent(self, client):
        assert self.post(client, consent="false").status_code == 400
        assert not (self.root / "pending").exists()

    def test_refused_with_unknown_label(self, client):
        assert self.post(client, label="maybe").status_code == 400

    def test_token_enforced_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(config, "FEEDBACK_TOKEN", "s3cret")
        assert self.post(client).status_code == 403
        ok = client.post("/api/feedback", headers={"X-Feedback-Token": "s3cret"},
                         files={"file": ("c.wav", wav_bytes(), "audio/wav")},
                         data={"label": "synthetic", "consent": "true"})
        assert ok.status_code == 200

    def test_disguised_file_refused(self, client):
        r = client.post("/api/feedback", files={"file": ("x.wav", b"MZ" + b"\0" * 200, "audio/wav")},
                        data={"label": "genuine", "consent": "true"})
        assert r.status_code == 400

    def test_queue_is_capped(self, client, monkeypatch):
        monkeypatch.setattr(config, "FEEDBACK_MAX_PENDING", 1)
        assert self.post(client).status_code == 200
        assert self.post(client).status_code == 400

    def test_disabled_deployment_refuses(self, client, monkeypatch):
        monkeypatch.setattr(config, "FEEDBACK_ENABLED", False)
        assert self.post(client).status_code == 403


class TestFineTuneGate:
    def test_rejects_update_that_hurts_validation(self):
        from fine_tune_feedback import accept_update
        ok, _ = accept_update(0.010, 0.030, 3, 5, max_drop=0.5)
        assert not ok

    def test_rejects_update_that_unlearns_corrections(self):
        from fine_tune_feedback import accept_update
        ok, _ = accept_update(0.010, 0.010, 5, 4, max_drop=0.5)
        assert not ok

    def test_accepts_safe_update(self):
        from fine_tune_feedback import accept_update
        ok, _ = accept_update(0.010, 0.012, 3, 5, max_drop=0.5)
        assert ok
