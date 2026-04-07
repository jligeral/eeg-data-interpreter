"""
Integration tests: run the full preprocess() pipeline on generated EEG files.
Generate the files first: python preprocessor/data/generate_test_files.py

Outputs are saved to preprocessor/tests/output/ as JSON and .txt summaries.
"""

import json
import pytest
from pathlib import Path
from preprocessor.preprocess import preprocess

GENERATED = Path(__file__).parent.parent / "data" / "generated"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def require_generated(filename):
    path = GENERATED / filename
    if not path.exists():
        pytest.skip(f"Generated file not found: {filename}. Run generate_test_files.py first.")
    return str(path)


def save_output(record, name):
    """Save JSON and plain-text summary of a preprocessed record."""
    (OUTPUT_DIR / f"{name}.json").write_text(
        json.dumps(record.to_dict(), indent=2)
    )
    (OUTPUT_DIR / f"{name}.txt").write_text(record.summarize())


class TestIntegration:

    def test_clean_full_montage_fif(self):
        record = preprocess(require_generated("clean_full_montage.fif"))
        save_output(record, "clean_full_montage")
        assert record.data_quality == "good"
        assert record.missing_channels == []
        assert record.confounds == []
        assert record.raw is not None

    def test_missing_posterior_edf(self):
        record = preprocess(require_generated("missing_posterior.edf"))
        save_output(record, "missing_posterior")
        assert record.data_quality == "poor"
        assert any("missing_posterior_channels" in c for c in record.confounds)
        assert "O1" in record.missing_channels or "O2" in record.missing_channels

    def test_flat_channel_fif(self):
        record = preprocess(require_generated("flat_channel.fif"))
        save_output(record, "flat_channel")
        assert len(record.flat_channels) >= 1
        assert any("flat_channels" in c for c in record.confounds)
        assert record.data_quality in ("moderate", "poor")

    def test_muscle_artifact_fif(self):
        record = preprocess(require_generated("muscle_artifact.fif"))
        save_output(record, "muscle_artifact")
        assert any("muscle_" in c for c in record.confounds)

    def test_drowsy_fif(self):
        record = preprocess(require_generated("drowsy.fif"))
        save_output(record, "drowsy")
        assert any("drowsiness_" in c for c in record.confounds)

    def test_low_sfreq_edf(self):
        record = preprocess(require_generated("low_sfreq.edf"))
        save_output(record, "low_sfreq")
        assert record.data_quality == "poor"
        assert any("sampling_rate_low" in c for c in record.confounds)
        assert record.sampling_rate == 64.0

    def test_channel_aliases_set(self):
        record = preprocess(require_generated("channel_aliases.set"))
        save_output(record, "channel_aliases")
        for ch in ("T3", "T4", "T5", "T6"):
            assert ch not in record.missing_channels

    def test_csv_with_sfreq(self):
        record = preprocess(require_generated("clean_full_montage.csv"), csv_sfreq=256.0)
        save_output(record, "clean_full_montage_csv")
        assert record.raw is not None
        assert record.sampling_rate == 256.0

    def test_processing_notes_always_populated(self):
        record = preprocess(require_generated("clean_full_montage.fif"))
        assert len(record.processing_notes) > 0

    def test_bad_file_returns_poor_record(self):
        record = preprocess("nonexistent_file.edf")
        save_output(record, "bad_file")
        assert record.data_quality == "poor"
        assert any("Load failed" in note for note in record.processing_notes)
