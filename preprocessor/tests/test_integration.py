"""
Integration tests: run the full preprocess() pipeline on generated EEG files.
Generate the files first: python preprocessor/data/generate_test_files.py
"""

import pytest
from pathlib import Path

from preprocessor.preprocess import preprocess

GENERATED = Path(__file__).parent.parent / "data" / "generated"


def require(filename):
    path = GENERATED / filename
    if not path.exists():
        pytest.skip(f"Generated file not found: {filename}. Run generate_test_files.py first.")
    return str(path)


class TestPreprocessIntegration:

    # --- Format coverage ---

    def test_fif(self):
        r = preprocess(require("clean_full_montage.fif"))
        assert r.data_quality in ("good", "moderate")  # ICA may flag mild artifacts in synthetic data
        assert r.referenced is True

    def test_edf(self):
        r = preprocess(require("clean_full_montage.edf"))
        assert r.data_quality in ("good", "moderate")
        assert r.referenced is True

    def test_set(self):
        r = preprocess(require("clean_full_montage.set"))
        assert r.data_quality in ("good", "moderate")

    def test_vhdr(self):
        r = preprocess(require("clean_full_montage.vhdr"))
        assert r.data_quality in ("good", "moderate")

    def test_csv(self):
        r = preprocess(require("clean_full_montage.csv"), sfreq=256)
        assert r.data_quality in ("good", "moderate")
        assert r.sampling_rate == 256.0

    def test_npz(self):
        r = preprocess(require("clean_full_montage.npz"))
        assert r.data_quality in ("good", "moderate")
        assert r.sampling_rate == 256.0

    def test_mat(self):
        r = preprocess(require("clean_full_montage.mat"))
        assert r.data_quality in ("good", "moderate")
        assert r.sampling_rate == 256.0

    # --- Pathological cases ---

    def test_missing_posterior_is_poor(self):
        r = preprocess(require("missing_posterior.edf"))
        assert r.data_quality == "poor"
        assert any("missing_posterior" in c for c in r.confounds)

    def test_flat_channel_detected(self):
        r = preprocess(require("flat_channel.fif"))
        assert any("flat_channels" in c for c in r.confounds)

    def test_muscle_artifact_detected(self):
        r = preprocess(require("muscle_artifact.fif"))
        assert any("muscle_" in c for c in r.confounds)

    def test_drowsy_detected(self):
        r = preprocess(require("drowsy.fif"))
        assert any("drowsiness_" in c for c in r.confounds)

    def test_low_sfreq_is_poor(self):
        r = preprocess(require("low_sfreq.edf"))
        assert r.data_quality == "poor"
        assert any("sampling_rate_low" in c for c in r.confounds)

    def test_channel_aliases_resolved(self):
        r = preprocess(require("channel_aliases.set"))
        assert "T3" in r.channel_labels
        assert "T7" not in r.channel_labels

    # --- Pipeline invariants ---

    def test_average_reference_always_applied(self):
        for fname in ("clean_full_montage.fif", "clean_full_montage.edf"):
            r = preprocess(require(fname))
            assert r.referenced is True, f"reference not applied for {fname}"

    def test_bandpass_note_present(self):
        r = preprocess(require("clean_full_montage.fif"))
        assert any("Bandpass" in n for n in r.processing_notes)

    def test_to_dict_serializable(self):
        import json
        r = preprocess(require("clean_full_montage.fif"))
        json.dumps(r.to_dict())

    def test_legacy_csv_sfreq_kwarg(self):
        r = preprocess(require("clean_full_montage.csv"), csv_sfreq=256)
        assert r.sampling_rate == 256.0
