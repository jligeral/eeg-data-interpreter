import numpy as np
import pytest
import mne
from unittest.mock import patch, MagicMock

from preprocessor.eeg_record import EEGRecord
from preprocessor.channel_validation import EXPECTED_CHANNELS_10_20
from feature_extractor.feature_record import FeatureRecord
from feature_extractor.resampler import resample_and_window, REVE_SFREQ, REVE_WINDOW
from feature_extractor.band_features import compute_band_features
from feature_extractor.reve_encoder import encode_with_reve
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
    # Add 10 Hz alpha to posterior channels
    for ch in ["O1", "O2", "P3", "P4"]:
        if ch in ch_names:
            idx = ch_names.index(ch)
            data[idx] += 25e-6 * np.sin(2 * np.pi * 10 * t)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    record = EEGRecord(
        raw=raw, filepath="synthetic", file_format=".fif",
        sampling_rate=sfreq, channel_labels=ch_names,
        data_quality="good",
    )
    return record


# ---------------------------------------------------------------------------
# Resampler tests
# ---------------------------------------------------------------------------

class TestResampler:

    def test_resamples_to_200hz(self):
        record = make_eeg_record(sfreq=256)
        windows, _, notes = resample_and_window(record)
        assert any(f"{REVE_SFREQ} Hz" in n for n in notes)

    def test_no_resample_needed_at_200hz(self):
        record = make_eeg_record(sfreq=200)
        windows, _, notes = resample_and_window(record)
        assert any("no resampling needed" in n.lower() for n in notes)

    def test_output_window_shape(self):
        record = make_eeg_record(sfreq=256, duration=30)
        windows, ch_names, _ = resample_and_window(record)
        n_windows = (30 * REVE_SFREQ) // REVE_WINDOW  # 6 windows from 30s
        assert windows.shape == (n_windows, len(EXPECTED_CHANNELS_10_20), REVE_WINDOW)

    def test_windows_are_normalized(self):
        record = make_eeg_record(sfreq=256, duration=30)
        windows, _, _ = resample_and_window(record)
        # Each window should be z-scored: mean ≈ 0, std ≈ 1, clipped to ±15
        assert np.abs(windows).max() <= 15.0
        means = windows.mean(axis=2)
        assert np.allclose(means, 0, atol=0.1)

    def test_short_recording_returns_empty(self):
        record = make_eeg_record(sfreq=200, duration=5)  # only 5s, need 10s
        windows, _, notes = resample_and_window(record)
        assert windows.size == 0
        assert any("too short" in n.lower() for n in notes)

    def test_no_raw_returns_empty(self):
        record = EEGRecord()
        windows, _, notes = resample_and_window(record)
        assert windows.size == 0
        assert any("skipped" in n for n in notes)

    def test_channel_names_returned(self):
        record = make_eeg_record(sfreq=256, duration=30)
        _, ch_names, _ = resample_and_window(record)
        assert len(ch_names) == len(EXPECTED_CHANNELS_10_20)

    def test_dtype_is_float32(self):
        record = make_eeg_record(sfreq=256, duration=30)
        windows, _, _ = resample_and_window(record)
        assert windows.dtype == np.float32


# ---------------------------------------------------------------------------
# Band feature tests
# ---------------------------------------------------------------------------

class TestBandFeatures:

    def test_all_bands_present(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        for band in ("delta", "theta", "alpha", "beta"):
            assert band in result["band_power_relative"]
            assert band in result["band_power_absolute"]

    def test_relative_power_sums_to_approx_one_per_region(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        rel = result["band_power_relative"]
        regions = set(r for b in rel.values() for r in b.keys())
        for region in regions:
            total = sum(rel[b].get(region, 0) for b in rel)
            # Sum of relative band powers won't be exactly 1 (only 4 bands shown)
            # but should be between 0 and 1
            assert 0.0 <= total <= 1.01

    def test_alpha_peak_detected_at_10hz(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        assert result["alpha_peak_frequency"] is not None
        # With 10 Hz alpha injected, peak should be close to 10 Hz
        assert 8.0 <= result["alpha_peak_frequency"] <= 13.0

    def test_alpha_peak_slowed_flagged(self):
        # Build record with 6 Hz "alpha" (slow/drowsy)
        ch_names = list(EXPECTED_CHANNELS_10_20)
        sfreq = 256
        duration = 30
        n = int(sfreq * duration)
        t = np.linspace(0, duration, n)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((len(ch_names), n)) * 5e-6
        for ch in ["O1", "O2"]:
            idx = ch_names.index(ch)
            data[idx] += 30e-6 * np.sin(2 * np.pi * 6 * t)  # 6 Hz
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
        result = compute_band_features(record)
        assert any("below 8 Hz" in n for n in result["notes"])

    def test_theta_alpha_ratio_computed(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        assert result["theta_alpha_ratio"] is not None
        assert result["theta_alpha_ratio"] > 0

    def test_posterior_alpha_power_computed(self):
        record = make_eeg_record()
        result = compute_band_features(record)
        assert result["posterior_alpha_power"] is not None
        assert 0.0 < result["posterior_alpha_power"] < 1.0

    def test_no_raw_returns_empty(self):
        record = EEGRecord()
        result = compute_band_features(record)
        assert result["band_power_relative"] == {}
        assert result["alpha_peak_frequency"] is None
        assert any("skipped" in n for n in result["notes"])

    def test_no_posterior_channels_skips_alpha_peak(self):
        ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "Cz", "C4"]
        record = make_eeg_record(ch_names=ch_names)
        result = compute_band_features(record)
        assert result["alpha_peak_frequency"] is None
        assert any("posterior" in n.lower() for n in result["notes"])


# ---------------------------------------------------------------------------
# REVE encoder tests (mocked)
# ---------------------------------------------------------------------------

class TestREVEEncoder:

    def _make_windows(self, n_windows=3, n_channels=19):
        return np.random.randn(n_windows, n_channels, REVE_WINDOW).astype(np.float32)

    def test_skips_when_windows_empty(self):
        feat = FeatureRecord()
        result = encode_with_reve(np.empty((0,), dtype=np.float32), [], feat)
        assert result.embeddings_available is False
        assert any("skipped" in n for n in result.processing_notes)

    def test_skips_gracefully_when_torch_missing(self):
        windows = self._make_windows()
        feat = FeatureRecord()
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            result = encode_with_reve(windows, list(EXPECTED_CHANNELS_10_20), feat)
        assert result.embeddings_available is False

    def test_skips_gracefully_when_model_unavailable(self):
        windows = self._make_windows()
        feat = FeatureRecord()
        with patch("feature_extractor.reve_encoder.AutoModel") as mock_automodel, \
             patch("feature_extractor.reve_encoder.torch", MagicMock()):
            mock_automodel.from_pretrained.side_effect = OSError("model not found")
            result = encode_with_reve(windows, list(EXPECTED_CHANNELS_10_20), feat)
        assert result.embeddings_available is False
        assert any("load failed" in n.lower() for n in result.processing_notes)

    def test_embeddings_shape_when_model_succeeds(self):
        n_windows, n_channels = 3, 19
        windows = self._make_windows(n_windows, n_channels)
        feat = FeatureRecord()

        import torch as real_torch

        # Real tensor simulating REVE's actual (W, C, H, 512) output shape
        fake_output = real_torch.zeros(n_windows, n_channels, 11, 512)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = fake_output

        mock_pos_bank = MagicMock()
        mock_pos_bank.to.return_value = mock_pos_bank
        mock_pos_bank.eval.return_value = mock_pos_bank
        mock_positions = MagicMock()
        mock_positions.unsqueeze.return_value.expand.return_value = mock_positions
        mock_pos_bank.return_value = mock_positions

        with patch("feature_extractor.reve_encoder.AutoModel") as mock_automodel, \
             patch("feature_extractor.reve_encoder.torch", real_torch):
            mock_automodel.from_pretrained.side_effect = [mock_pos_bank, mock_model]

            result = encode_with_reve(windows, list(EXPECTED_CHANNELS_10_20), feat)

        assert result.embeddings_available is True
        assert result.embeddings.shape == (n_windows, 512)
        assert result.n_windows == n_windows


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestExtractPipeline:

    def test_extract_returns_feature_record(self):
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

    def test_processing_notes_include_all_steps(self):
        record = make_eeg_record()
        feat = extract(record)
        combined = " ".join(feat.processing_notes).lower()
        assert "band features" in combined
        assert "hz" in combined  # resampling note

    def test_no_raw_degrades_gracefully(self):
        record = EEGRecord(data_quality="poor")
        feat = extract(record)
        assert isinstance(feat, FeatureRecord)
        assert feat.band_power_relative == {}
        assert feat.embeddings_available is False

    def test_to_dict_is_serializable(self):
        import json
        record = make_eeg_record()
        feat = extract(record)
        d = feat.to_dict()
        json.dumps(d)  # should not raise

    def test_summarize_output(self):
        record = make_eeg_record()
        feat = extract(record)
        summary = feat.summarize()
        assert "Feature Record Summary" in summary
        assert "alpha" in summary.lower() or "Alpha" in summary
