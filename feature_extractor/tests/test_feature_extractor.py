import numpy as np
import pytest
import mne
from unittest.mock import patch, MagicMock

from preprocessor.eeg_record import EEGRecord
from preprocessor.channel_validation import EXPECTED_CHANNELS_10_20
from feature_extractor.feature_record import FeatureRecord
from feature_extractor.windower import (
    resample_and_window, LEAD_SFREQ, LEAD_WINDOW, LEAD_CHANNEL_ORDER
)
from feature_extractor.band_features import compute_band_features, compute_posterior_coherence
from feature_extractor.lead_encoder import encode_with_lead
from feature_extractor.extract import extract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eeg_record(ch_names=None, sfreq=256, duration=30, seed=42):
    if ch_names is None:
        ch_names = list(EXPECTED_CHANNELS_10_20)
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    t = np.linspace(0, duration, n)
    data = rng.standard_normal((len(ch_names), n)) * 10e-6
    for ch in ["O1", "O2", "P3", "P4"]:
        if ch in ch_names:
            data[ch_names.index(ch)] += 25e-6 * np.sin(2 * np.pi * 10 * t)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return EEGRecord(
        raw=raw, filepath="synthetic", file_format=".fif",
        sampling_rate=sfreq, channel_labels=ch_names,
        data_quality="good",
    )


# ---------------------------------------------------------------------------
# Windower
# ---------------------------------------------------------------------------

class TestWindower:

    def test_resamples_to_200hz(self):
        record = make_eeg_record(sfreq=256)
        windows, ch_names, notes = resample_and_window(record)
        assert any(f"{LEAD_SFREQ} Hz" in n for n in notes)

    def test_no_resample_at_200hz(self):
        record = make_eeg_record(sfreq=200)
        _, _, notes = resample_and_window(record)
        assert any("no resampling" in n.lower() for n in notes)

    def test_output_shape_is_b_t_c(self):
        record = make_eeg_record(sfreq=256, duration=30)
        windows, ch_names, _ = resample_and_window(record)
        n_windows = (30 * LEAD_SFREQ) // LEAD_WINDOW
        assert windows.shape == (n_windows, LEAD_WINDOW, len(ch_names))

    def test_channel_order_matches_lead(self):
        record = make_eeg_record()
        _, ch_names, _ = resample_and_window(record)
        # Returned channels should be a prefix of LEAD_CHANNEL_ORDER
        for ch in ch_names:
            assert ch in LEAD_CHANNEL_ORDER

    def test_windows_normalized(self):
        record = make_eeg_record(sfreq=256, duration=30)
        windows, _, _ = resample_and_window(record)
        assert np.abs(windows).max() <= 10.0
        means = windows.mean(axis=1)
        assert np.allclose(means, 0, atol=0.1)

    def test_short_recording_returns_empty(self):
        record = make_eeg_record(sfreq=200, duration=1)
        windows, _, notes = resample_and_window(record)
        assert windows.size == 0
        assert any("too short" in n.lower() for n in notes)

    def test_no_raw_returns_empty(self):
        record = EEGRecord()
        windows, _, notes = resample_and_window(record)
        assert windows.size == 0
        assert any("skipped" in n for n in notes)

    def test_dtype_is_float32(self):
        record = make_eeg_record()
        windows, _, _ = resample_and_window(record)
        assert windows.dtype == np.float32


# ---------------------------------------------------------------------------
# Band features
# ---------------------------------------------------------------------------

class TestBandFeatures:

    def test_all_bands_present(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        for band in ("delta", "theta", "alpha", "beta"):
            assert band in result["band_power_relative"]

    def test_alpha_peak_near_10hz(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        assert result["alpha_peak_frequency"] is not None
        assert 8.0 <= result["alpha_peak_frequency"] <= 13.0

    def test_slowed_alpha_flagged(self):
        ch_names = list(EXPECTED_CHANNELS_10_20)
        sfreq, duration = 256, 30
        n = int(sfreq * duration)
        t = np.linspace(0, duration, n)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((len(ch_names), n)) * 5e-6
        for ch in ["O1", "O2"]:
            data[ch_names.index(ch)] += 30e-6 * np.sin(2 * np.pi * 6 * t)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        record = EEGRecord(raw=raw, filepath="s", file_format=".fif")
        result = compute_band_features(record)
        assert any("below 8 Hz" in n for n in result["notes"])

    def test_theta_alpha_ratio_positive(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        assert result["theta_alpha_ratio"] is not None
        assert result["theta_alpha_ratio"] > 0

    def test_no_raw_returns_empty(self):
        record = EEGRecord()
        result = compute_band_features(record)
        assert result["band_power_relative"] == {}
        assert result["alpha_peak_frequency"] is None


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

class TestPosteriorCoherence:

    def test_coherence_computed(self):
        record = make_eeg_record()
        result = compute_posterior_coherence(record)
        assert result["posterior_alpha_coherence"] is not None
        assert 0.0 <= result["posterior_alpha_coherence"] <= 1.0

    def test_coherence_pairs_returned(self):
        record = make_eeg_record()
        result = compute_posterior_coherence(record)
        assert len(result["coherence_pairs"]) > 0

    def test_skips_without_posterior_channels(self):
        ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "Cz", "C4"]
        record = make_eeg_record(ch_names=ch_names)
        result = compute_posterior_coherence(record)
        assert result["posterior_alpha_coherence"] is None
        assert any("skipped" in n.lower() for n in result["notes"])

    def test_skips_without_raw(self):
        result = compute_posterior_coherence(EEGRecord())
        assert result["posterior_alpha_coherence"] is None


# ---------------------------------------------------------------------------
# LEAD encoder (mocked)
# ---------------------------------------------------------------------------

class TestLEADEncoder:

    def _make_windows(self, n_windows=3, n_channels=19):
        return np.random.randn(n_windows, LEAD_WINDOW, n_channels).astype(np.float32)

    def test_skips_when_windows_empty(self):
        feat = FeatureRecord()
        result = encode_with_lead(np.empty((0,), dtype=np.float32), [], feat)
        assert result.lead_embeddings_available is False
        assert any("skipped" in n for n in result.processing_notes)

    def test_skips_when_torch_missing(self):
        windows = self._make_windows()
        feat = FeatureRecord()
        with patch("feature_extractor.lead_encoder._try_load_deps", return_value=(False, "torch not installed")):
            result = encode_with_lead(windows, list(LEAD_CHANNEL_ORDER), feat)
        assert result.lead_embeddings_available is False

    def test_skips_when_lead_source_missing(self):
        windows = self._make_windows()
        feat = FeatureRecord()
        with patch("feature_extractor.lead_encoder._try_load_deps", return_value=(False, "LEAD source not found")):
            result = encode_with_lead(windows, list(LEAD_CHANNEL_ORDER), feat)
        assert result.lead_embeddings_available is False
        assert any("LEAD encoding skipped" in n for n in result.processing_notes)

    def test_embeddings_shape_when_model_succeeds(self):
        import torch as real_torch

        n_windows, n_channels, d_model = 3, 19, 128
        windows = self._make_windows(n_windows, n_channels)
        feat = FeatureRecord()

        fake_logits = real_torch.zeros(n_windows, 2)
        fake_embedding = real_torch.zeros(n_windows, n_channels * d_model)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = fake_logits

        # Simulate the projection_head hook: hook captures fake_embedding
        # pretrain_lead models use projection_head; forward returns a tuple
        mock_model.return_value = (real_torch.zeros(n_windows, 128), real_torch.zeros(n_windows, 128))

        captured_hook_fn = None
        def fake_register_hook(fn):
            nonlocal captured_hook_fn
            captured_hook_fn = fn
            # Immediately fire the hook with fake embedding as input
            fn(mock_model.projection_head, (fake_embedding,), fake_logits)
            return MagicMock()  # hook handle

        mock_model.projection_head.register_forward_hook = fake_register_hook

        with patch("feature_extractor.lead_encoder._try_load_deps", return_value=(True, "")), \
             patch("feature_extractor.lead_encoder._load_model", return_value=(mock_model, False, "")), \
             patch("feature_extractor.lead_encoder._torch", real_torch):
            result = encode_with_lead(windows, list(LEAD_CHANNEL_ORDER[:n_channels]), feat)

        assert result.lead_embeddings_available is True
        assert result.lead_embeddings.shape == (n_windows, n_channels * d_model)
        assert result.n_windows == n_windows

    def test_ad_probability_populated_when_classifier_available(self):
        import torch as real_torch

        n_windows, n_channels, d_model = 3, 19, 128
        windows = self._make_windows(n_windows, n_channels)
        feat = FeatureRecord()

        # Logits strongly favouring AD class (index 1)
        fake_logits = real_torch.tensor([[0.1, 5.0]] * n_windows)
        fake_embedding = real_torch.zeros(n_windows, n_channels * d_model)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = fake_logits

        # Fine-tuned model uses classifier layer; forward returns logits directly
        def fake_register_hook(fn):
            fn(mock_model.classifier, (fake_embedding,), fake_logits)
            return MagicMock()

        mock_model.classifier.register_forward_hook = fake_register_hook
        # hasattr checks: finetune model has classifier, not projection_head
        del mock_model.projection_head

        with patch("feature_extractor.lead_encoder._try_load_deps", return_value=(True, "")), \
             patch("feature_extractor.lead_encoder._load_model", return_value=(mock_model, True, "")), \
             patch("feature_extractor.lead_encoder._torch", real_torch):
            result = encode_with_lead(windows, list(LEAD_CHANNEL_ORDER[:n_channels]), feat)

        assert result.lead_ad_probability is not None
        assert result.lead_ad_probability > 0.9
        assert len(result.lead_ad_per_window) == n_windows


# ---------------------------------------------------------------------------
# Full extract pipeline
# ---------------------------------------------------------------------------

class TestExtractPipeline:

    def test_returns_feature_record(self):
        record = make_eeg_record()
        feat = extract(record)
        assert isinstance(feat, FeatureRecord)

    def test_metadata_passed_through(self):
        record = make_eeg_record()
        record.data_quality = "moderate"
        record.confounds = ["muscle_mild:temporal"]
        feat = extract(record)
        assert feat.data_quality == "moderate"
        assert "muscle_mild:temporal" in feat.confounds

    def test_band_features_populated(self):
        record = make_eeg_record()
        feat = extract(record)
        assert feat.band_power_relative != {}
        assert feat.alpha_peak_frequency is not None
        assert feat.theta_alpha_ratio is not None

    def test_coherence_populated(self):
        record = make_eeg_record()
        feat = extract(record)
        assert feat.posterior_alpha_coherence is not None

    def test_graceful_degradation_no_raw(self):
        record = EEGRecord(data_quality="poor")
        feat = extract(record)
        assert isinstance(feat, FeatureRecord)
        assert feat.band_power_relative == {}
        assert feat.lead_embeddings_available is False

    def test_to_dict_serializable(self):
        import json
        record = make_eeg_record()
        feat = extract(record)
        json.dumps(feat.to_dict())
