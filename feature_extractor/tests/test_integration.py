"""
Integration tests: preprocess() → extract() on generated EEG files.
Generate the files first: python preprocessor/data/generate_test_files.py
"""

import pytest
from pathlib import Path

from preprocessor.preprocess import preprocess
from feature_extractor.extract import extract

GENERATED = Path(__file__).parent.parent.parent / "preprocessor" / "data" / "generated"


def require(filename):
    path = GENERATED / filename
    if not path.exists():
        pytest.skip(f"Generated file not found: {filename}. Run generate_test_files.py first.")
    return str(path)


class TestFeatureExtractorIntegration:

    def test_clean_fif(self):
        feat = extract(preprocess(require("clean_full_montage.fif")))
        assert feat.band_power_relative != {}
        assert feat.alpha_peak_frequency is not None
        assert feat.theta_alpha_ratio is not None
        assert feat.posterior_alpha_coherence is not None

    def test_clean_edf(self):
        feat = extract(preprocess(require("clean_full_montage.edf")))
        assert feat.band_power_relative != {}

    def test_clean_npz(self):
        feat = extract(preprocess(require("clean_full_montage.npz")))
        assert feat.alpha_peak_frequency is not None

    def test_missing_posterior(self):
        feat = extract(preprocess(require("missing_posterior.edf")))
        assert feat.data_quality == "poor"
        assert any("missing_posterior" in c for c in feat.confounds)

    def test_drowsy_has_elevated_theta_alpha(self):
        feat_clean = extract(preprocess(require("clean_full_montage.fif")))
        feat_drowsy = extract(preprocess(require("drowsy.fif")))
        assert feat_drowsy.theta_alpha_ratio > feat_clean.theta_alpha_ratio

    def test_channel_aliases_resolved(self):
        feat = extract(preprocess(require("channel_aliases.set")))
        assert feat.band_power_relative != {}
        assert feat.alpha_peak_frequency is not None

    def test_windower_note_present(self):
        feat = extract(preprocess(require("clean_full_montage.fif")))
        assert any("window" in n.lower() for n in feat.processing_notes)

    def test_to_dict_serializable(self):
        import json
        feat = extract(preprocess(require("clean_full_montage.fif")))
        json.dumps(feat.to_dict())

    def test_preprocessor_notes_carried_through(self):
        feat = extract(preprocess(require("clean_full_montage.fif")))
        assert len(feat.preprocessor_notes) > 0
        assert any("Bandpass" in n for n in feat.preprocessor_notes)
