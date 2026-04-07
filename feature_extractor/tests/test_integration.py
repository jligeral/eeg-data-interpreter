"""
Integration tests: run the full preprocess() → extract() pipeline on generated EEG files.
Generate the files first: python preprocessor/data/generate_test_files.py

Outputs are saved to feature_extractor/tests/output/ as JSON and .txt summaries.
"""

import json
import pytest
from pathlib import Path

from preprocessor.preprocess import preprocess
from feature_extractor.extract import extract

GENERATED = Path(__file__).parent.parent.parent / "preprocessor" / "data" / "generated"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def require_generated(filename):
    path = GENERATED / filename
    if not path.exists():
        pytest.skip(f"Generated file not found: {filename}. Run generate_test_files.py first.")
    return str(path)


def save_output(feat, name):
    (OUTPUT_DIR / f"{name}.json").write_text(
        json.dumps(feat.to_dict(), indent=2)
    )
    (OUTPUT_DIR / f"{name}.txt").write_text(feat.summarize())


class TestFeatureExtractorIntegration:

    def test_clean_full_montage(self):
        feat = extract(preprocess(require_generated("clean_full_montage.fif")))
        save_output(feat, "clean_full_montage")
        assert feat.data_quality == "good"
        assert feat.band_power_relative != {}
        assert feat.alpha_peak_frequency is not None
        assert feat.theta_alpha_ratio is not None
        assert feat.posterior_alpha_power is not None

    def test_missing_posterior(self):
        feat = extract(preprocess(require_generated("missing_posterior.edf")))
        save_output(feat, "missing_posterior")
        assert feat.data_quality == "poor"
        assert any("missing_posterior" in c for c in feat.confounds)
        # T5/T6/P3/P4 are still present so alpha peak can still be computed
        assert feat.band_power_relative != {}

    def test_flat_channel(self):
        feat = extract(preprocess(require_generated("flat_channel.fif")))
        save_output(feat, "flat_channel")
        assert any("flat_channels" in c for c in feat.confounds)
        assert feat.band_power_relative != {}

    def test_muscle_artifact(self):
        feat = extract(preprocess(require_generated("muscle_artifact.fif")))
        save_output(feat, "muscle_artifact")
        assert any("muscle_" in c for c in feat.confounds)
        assert feat.alpha_peak_frequency is not None

    def test_drowsy(self):
        feat = extract(preprocess(require_generated("drowsy.fif")))
        save_output(feat, "drowsy")
        assert any("drowsiness_" in c for c in feat.confounds)
        # Drowsy signal has slowing alpha — peak should be below normal (< 10 Hz)
        assert feat.alpha_peak_frequency is not None
        assert feat.alpha_peak_frequency < 10.0

    def test_low_sfreq(self):
        feat = extract(preprocess(require_generated("low_sfreq.edf")))
        save_output(feat, "low_sfreq")
        assert feat.data_quality == "poor"
        assert any("sampling_rate_low" in c for c in feat.confounds)
        # Band features should still be computed despite poor quality
        assert feat.band_power_relative != {}

    def test_channel_aliases(self):
        feat = extract(preprocess(require_generated("channel_aliases.set")))
        save_output(feat, "channel_aliases")
        assert feat.band_power_relative != {}
        assert feat.alpha_peak_frequency is not None

    def test_csv_with_sfreq(self):
        feat = extract(preprocess(require_generated("clean_full_montage.csv"), csv_sfreq=256.0))
        save_output(feat, "clean_full_montage_csv")
        assert feat.sampling_rate == 256.0
        assert feat.band_power_relative != {}

    def test_processing_notes_span_both_modules(self):
        feat = extract(preprocess(require_generated("clean_full_montage.fif")))
        combined = " ".join(feat.processing_notes + feat.preprocessor_notes).lower()
        assert "bandpass" in combined        # preprocessor note
        assert "band features" in combined   # feature extractor note
        assert "hz" in combined              # resampler note

    def test_to_dict_is_json_serializable(self):
        feat = extract(preprocess(require_generated("clean_full_montage.fif")))
        d = feat.to_dict()
        json.dumps(d)  # should not raise

    def test_theta_alpha_ratio_elevated_in_drowsy(self):
        feat_clean = extract(preprocess(require_generated("clean_full_montage.fif")))
        feat_drowsy = extract(preprocess(require_generated("drowsy.fif")))
        assert feat_drowsy.theta_alpha_ratio > feat_clean.theta_alpha_ratio
