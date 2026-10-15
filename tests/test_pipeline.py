"""Tests for the audio front-end, metrics and risk mapping."""

import numpy as np
import pytest
import soundfile as sf

import config
from backend.preprocessor import AudioPreprocessor
from backend.risk_engine import PROFILES, RiskScoringEngine
from model.evaluate import compute_eer, evaluate_scores, per_attack_eer


@pytest.fixture(scope="module")
def pre() -> AudioPreprocessor:
    return AudioPreprocessor()


def _tone(seconds: float, freq: float = 220.0, sr: int = config.SAMPLE_RATE):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(tmp_path, name, audio, sr=config.SAMPLE_RATE):
    path = tmp_path / name
    sf.write(path, audio, sr)
    return str(path)


class TestPreprocessor:
    def test_spectrogram_shape_and_range(self, pre, tmp_path):
        path = _write(tmp_path, "a.wav", _tone(3.0))
        _, batch, preview = pre.process(path)
        assert batch.shape == (1, 1, config.N_MELS, config.TARGET_FRAMES)
        assert preview.shape == (config.N_MELS, config.TARGET_FRAMES)
        assert 0.0 <= preview.min() and preview.max() <= 1.0

    def test_short_audio_is_padded_not_rejected(self, pre, tmp_path):
        path = _write(tmp_path, "short.wav", _tone(0.4))
        _, batch, _ = pre.process(path)
        assert batch.shape[0] == 1

    def test_long_audio_produces_multiple_windows(self, pre, tmp_path):
        """v1 truncated at 3 s and never looked at the rest of the recording."""
        path = _write(tmp_path, "long.wav", _tone(12.0))
        _, batch, _ = pre.process(path)
        assert batch.shape[0] >= 6

    def test_window_count_is_bounded(self, pre, tmp_path):
        path = _write(tmp_path, "verylong.wav", _tone(180.0))
        _, batch, _ = pre.process(path)
        assert batch.shape[0] <= config.MAX_WINDOWS_PER_CLIP

    def test_resamples_other_rates(self, pre, tmp_path):
        path = _write(tmp_path, "44k.wav", _tone(3.0, sr=44_100), sr=44_100)
        waveform, _, _ = pre.process(path)
        assert abs(len(waveform) / config.SAMPLE_RATE - 3.0) < 0.05

    def test_gain_does_not_change_the_feature(self, pre, tmp_path):
        """
        Peak normalisation must make the detector insensitive to recording
        level, so that a quiet caller is not scored differently from a loud
        one. Agreement is to within float32 quantisation of the stored WAV,
        not bit-exact.
        """
        loud = _write(tmp_path, "loud.wav", _tone(3.0) * 0.95)
        quiet = _write(tmp_path, "quiet.wav", _tone(3.0) * 0.05)
        _, _, a = pre.process(loud)
        _, _, b = pre.process(quiet)
        assert np.mean(np.abs(a - b)) < 1e-3
        assert np.max(np.abs(a - b)) < 0.02

    def test_rejects_over_long_audio(self, pre, tmp_path):
        path = _write(tmp_path, "huge.wav", _tone(config.MAX_DECODED_SECONDS + 5))
        with pytest.raises(ValueError, match="limit is"):
            pre.process(path)

    def test_rejects_undecodable_bytes(self, pre, tmp_path):
        path = tmp_path / "junk.wav"
        path.write_bytes(b"not audio at all")
        with pytest.raises(ValueError):
            pre.process(str(path))

    def test_training_and_serving_share_the_feature(self, pre, tmp_path):
        """A single 3 s clip must give the same feature on both code paths."""
        path = _write(tmp_path, "same.wav", _tone(3.0))
        _, batch, _ = pre.process(path)
        single = pre.process_single(path)
        assert np.allclose(batch[0], single, atol=1e-5)


class TestEER:
    def test_perfect_separation_is_zero(self):
        y = np.array([0] * 20 + [1] * 20)
        s = np.concatenate([np.full(20, 0.01), np.full(20, 0.99)])
        eer, _ = compute_eer(y, s)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_random_scores_are_near_half(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 500 + [1] * 500)
        eer, _ = compute_eer(y, rng.random(1000))
        assert 0.4 < eer < 0.6

    def test_threshold_is_finite(self):
        """sklearn prepends an infinite threshold; it must never be returned."""
        rng = np.random.default_rng(1)
        y = np.array([0] * 50 + [1] * 50)
        s = np.concatenate([rng.beta(1, 4, 50), rng.beta(4, 1, 50)])
        _, threshold = compute_eer(y, s)
        assert np.isfinite(threshold)
        assert 0.0 <= threshold <= 1.0

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            compute_eer(np.zeros(10), np.random.rand(10))

    def test_accuracy_is_not_the_headline_on_imbalance(self):
        """A constant 'spoof' answer scores 90% accuracy and 50% EER."""
        y = np.array([0] * 100 + [1] * 900)
        s = np.full(1000, 0.99)
        metrics = evaluate_scores(y, s)
        assert metrics["accuracy"] >= 0.89
        assert metrics["balanced_accuracy"] == pytest.approx(0.5, abs=0.01)

    def test_false_acceptance_is_the_clone_that_got_through(self):
        """
        FAR must count clones accepted, not genuine speakers refused.

        With positive = spoof, "false positive" means a genuine speaker was
        flagged, which is a false *rejection*. Naming these the wrong way round
        inverts the two numbers a reader weighs against each other, and the
        mistake is invisible unless it is asserted.
        """
        # 10 bonafide scoring 0.0, 10 spoof scoring 1.0, threshold above both
        # spoof and bonafide -> every clone is accepted, nobody is refused.
        y = np.array([0] * 10 + [1] * 10)
        s = np.array([0.0] * 10 + [0.4] * 10)
        m = evaluate_scores(y, s, threshold=0.9)
        assert m["false_acceptance_rate"] == pytest.approx(1.0)
        assert m["false_rejection_rate"] == pytest.approx(0.0)
        assert m["confusion_matrix"]["false_negative_clone_accepted"] == 10

        # Threshold below both -> everything is called spoof: every genuine
        # speaker is refused, no clone gets through.
        m = evaluate_scores(y, s, threshold=-0.1)
        assert m["false_acceptance_rate"] == pytest.approx(0.0)
        assert m["false_rejection_rate"] == pytest.approx(1.0)
        assert m["confusion_matrix"]["false_positive_genuine_rejected"] == 10

    def test_far_and_frr_meet_at_the_eer(self):
        """At the EER operating point the two error rates coincide."""
        rng = np.random.default_rng(3)
        y = np.array([0] * 400 + [1] * 400)
        s = np.concatenate([rng.beta(2, 5, 400), rng.beta(5, 2, 400)])
        eer, threshold = compute_eer(y, s)
        m = evaluate_scores(y, s, threshold=threshold)
        assert m["false_acceptance_rate"] == pytest.approx(eer, abs=0.02)
        assert m["false_rejection_rate"] == pytest.approx(eer, abs=0.02)

    def test_per_attack_breakdown(self):
        rng = np.random.default_rng(7)
        y = np.array([0] * 40 + [1] * 40)
        scores = np.concatenate([rng.beta(1, 6, 40), rng.beta(6, 1, 20), rng.beta(1, 3, 20)])
        attacks = np.array(["-"] * 40 + ["A01"] * 20 + ["A17"] * 20)
        breakdown = per_attack_eer(y, scores, attacks)
        assert set(breakdown) == {"A01", "A17"}
        assert breakdown["A17"]["eer_percent"] > breakdown["A01"]["eer_percent"]


class TestRiskEngine:
    THRESHOLD = 0.5

    def test_tiers_increase_with_probability(self):
        low = RiskScoringEngine.assess(0.05, self.THRESHOLD)
        med = RiskScoringEngine.assess(0.45, self.THRESHOLD)
        high = RiskScoringEngine.assess(0.95, self.THRESHOLD)
        assert (low.risk_tier, med.risk_tier, high.risk_tier) == ("LOW", "MEDIUM", "HIGH")

    def test_profiles_shift_the_operating_point(self):
        """
        The same score must be treated more severely where the loss is larger.
        A payment authorisation blocks at a probability where a support desk
        only asks a second question.
        """
        severity = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        prob = 0.40
        bank = RiskScoringEngine.assess(prob, self.THRESHOLD, "financial_services")
        desk = RiskScoringEngine.assess(prob, self.THRESHOLD, "contact_center")
        assert bank.risk_tier == "HIGH"
        assert severity[bank.risk_tier] > severity[desk.risk_tier]
        assert bank.action_code == "BLOCK_AND_ESCALATE"
        assert desk.action_code != "BLOCK_AND_ESCALATE"

    def test_unknown_profile_falls_back_to_generic(self):
        assessment = RiskScoringEngine.assess(0.9, self.THRESHOLD, "not_a_profile")
        assert assessment.profile == "generic"

    def test_tiers_track_a_calibrated_threshold(self):
        """A low calibrated threshold must make the same score more severe."""
        strict = RiskScoringEngine.assess(0.30, 0.20)
        lenient = RiskScoringEngine.assess(0.30, 0.90)
        assert strict.risk_tier == "HIGH"
        assert lenient.risk_tier == "LOW"

    def test_probability_is_clamped(self):
        assert RiskScoringEngine.assess(5.0, self.THRESHOLD).spoof_probability == 1.0
        assert RiskScoringEngine.assess(-3.0, self.THRESHOLD).spoof_probability == 0.0

    def test_caveats_survive_into_the_result(self):
        assessment = RiskScoringEngine.assess(
            0.9, self.THRESHOLD, caveats=["model is untrained"]
        )
        assert "model is untrained" in assessment.to_dict()["caveats"]

    def test_every_profile_is_well_formed(self):
        for key, profile in PROFILES.items():
            assert profile.review_ratio < profile.block_ratio
            assert profile.step_up_action and profile.block_action
            assert RiskScoringEngine.assess(0.5, 0.5, key).profile == key


class TestSilenceHandling:
    """The first model learned how much silence surrounded a clip. These pin the fix."""

    def test_leading_and_trailing_silence_is_removed(self, pre):
        sr = config.SAMPLE_RATE
        speech = _tone(1.0)
        y = np.concatenate([np.zeros(sr), speech, np.zeros(sr)]).astype(np.float32)
        trimmed = pre.trim_silence(y)
        assert abs(len(trimmed) / sr - 1.0) < 0.15

    def test_same_speech_gives_same_feature_whatever_the_silence(self, pre, tmp_path):
        sr = config.SAMPLE_RATE
        speech = _tone(2.0, 300.0)
        a = _write(tmp_path, "tight.wav", speech)
        b = _write(tmp_path, "padded.wav", np.concatenate([np.zeros(int(1.5 * sr), np.float32), speech]))
        assert np.mean(np.abs(pre.process_single(a) - pre.process_single(b))) < 0.01

    def test_short_clips_are_repeated_not_zero_padded(self, pre):
        y = _tone(0.8)
        out = pre.fit_length(y, pre.window_samples)
        assert len(out) == pre.window_samples
        assert np.mean(np.abs(out[-1000:])) > 0.05          # no silent tail

    def test_near_silent_windows_are_dropped(self, pre):
        sr = config.SAMPLE_RATE
        y = np.concatenate([_tone(6.0), np.full(6 * sr, 1e-5, np.float32)])
        assert len(pre.split_windows(y)) < len(pre.split_windows(_tone(12.0)))

    def test_very_short_clip_is_not_trimmed_away(self, pre):
        y = _tone(0.3)
        assert len(pre.trim_silence(y)) == len(y)


class TestAugmentationAndVocoder:
    def test_channels_keep_length_and_are_finite(self):
        from model import augment
        rng = np.random.default_rng(0)
        y = _tone(2.0)
        for kind in ("telephone", "g711", "noise", "noise20", "lowpass", "clean"):
            out = augment.apply(y, kind, rng)
            assert abs(len(out) - len(y)) <= 2 and np.all(np.isfinite(out))

    def test_telephone_removes_high_frequencies(self):
        from model import augment
        y = _tone(1.0, 6000.0)
        assert np.sqrt(np.mean(augment.telephone(y) ** 2)) < 0.05 * np.sqrt(np.mean(y ** 2))

    def test_vocoder_returns_normalised_audio_of_similar_length(self):
        from model import vocode
        y = _tone(1.5, 200.0)
        for variant in (0, 1):
            out = vocode.resynthesise(y, variant)
            assert abs(len(out) - len(y)) < 600
            assert 0.99 <= np.max(np.abs(out)) <= 1.0
