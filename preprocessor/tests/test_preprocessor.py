import io
import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest
import scipy.io

from preprocessor.eeg_record import EEGRecord
from preprocessor.channel_validation import (
    EXPECTED_CHANNELS_10_20,
    canonicalize_channel,
    validate_channels,
)
from preprocessor.bandpass_filter import apply_bandpass_filter, HIGH_CUTOFF_HZ, LOW_CUTOFF_HZ
from preprocessor.reference import apply_average_reference
from preprocessor.artifact_detection import detect_artifacts
from preprocessor.format_adapter import (
    UnsupportedFormatError,
    CorruptFileError,
    load_eeg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw(ch_names=None, sfreq=256, duration=30, seed=42):
    if ch_names is None:
        ch_names = list(EXPECTED_CHANNELS_10_20)
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    t = np.linspace(0, duration, n)
    data = rng.standard_normal((len(ch_names), n)) * 10e-6
    for ch in ["O1", "O2"]:
        if ch in ch_names:
            data[ch_names.index(ch)] += 25e-6 * np.sin(2 * np.pi * 10 * t)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def make_record(ch_names=None, sfreq=256, duration=30, seed=42):
    raw = make_raw(ch_names=ch_names, sfreq=sfreq, duration=duration, seed=seed)
    return EEGRecord(raw=raw, filepath="synthetic.fif", file_format=".fif")


# ---------------------------------------------------------------------------
# Channel validation
# ---------------------------------------------------------------------------

class TestChannelValidation:

    def test_canonicalize_aliases(self):
        assert canonicalize_channel("T7") == "T3"
        assert canonicalize_channel("T8") == "T4"
        assert canonicalize_channel("P7") == "T5"
        assert canonicalize_channel("P8") == "T6"

    def test_canonicalize_strips_suffixes(self):
        assert canonicalize_channel("Fp1-REF") == "Fp1"
        assert canonicalize_channel("EEG Fz") == "Fz"
        assert canonicalize_channel("O1-LE") == "O1"

    def test_full_montage_is_good(self):
        record = make_record()
        record = validate_channels(record)
        assert record.data_quality == "good"
        assert record.missing_channels == []

    def test_missing_posterior_is_poor(self):
        ch = [c for c in EXPECTED_CHANNELS_10_20 if c not in ("O1", "O2")]
        record = make_record(ch_names=ch)
        record = validate_channels(record)
        assert record.data_quality == "poor"
        assert any("missing_posterior" in c for c in record.confounds)

    def test_flat_channel_detected(self):
        record = make_record()
        record.raw._data[0, :] = 0.0
        record = validate_channels(record)
        assert len(record.flat_channels) >= 1
        assert any("flat_channels" in c for c in record.confounds)

    def test_high_amplitude_detected(self):
        record = make_record()
        record.raw._data[0, :] = 600e-6
        record = validate_channels(record)
        assert len(record.high_amplitude_channels) >= 1

    def test_low_sfreq_is_poor(self):
        record = make_record(sfreq=64)
        record = validate_channels(record)
        assert record.data_quality == "poor"
        assert any("sampling_rate_low" in c for c in record.confounds)

    def test_alias_channels_renamed(self):
        aliased = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T7",  "C3",  "Cz", "C4", "T8",
            "P7",  "P3",  "Pz", "P4", "P8",
            "O1",  "O2",
        ]
        record = make_record(ch_names=aliased)
        record = validate_channels(record)
        assert "T3" in record.channel_labels
        assert "T7" not in record.channel_labels

    def test_no_raw_returns_poor(self):
        record = EEGRecord()
        record = validate_channels(record)
        assert record.data_quality == "poor"


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:

    def test_filter_applied(self):
        record = make_record()
        record = apply_bandpass_filter(record)
        assert any("Bandpass" in n for n in record.processing_notes)

    def test_correct_cutoffs_in_notes(self):
        record = make_record()
        record = apply_bandpass_filter(record)
        notes = " ".join(record.processing_notes)
        assert str(LOW_CUTOFF_HZ) in notes
        assert "45" in notes

    def test_high_cutoff_adjusted_for_low_sfreq(self):
        record = make_record(sfreq=64)
        record = apply_bandpass_filter(record)
        assert any("adjusted" in n for n in record.processing_notes)

    def test_no_raw_skips(self):
        record = EEGRecord()
        record = apply_bandpass_filter(record)
        assert any("skipped" in n for n in record.processing_notes)


# ---------------------------------------------------------------------------
# Average reference
# ---------------------------------------------------------------------------

class TestAverageReference:

    def test_reference_applied(self):
        record = make_record()
        record = apply_average_reference(record)
        assert record.referenced is True
        assert any("Average reference" in n for n in record.processing_notes)

    def test_channel_means_near_zero(self):
        record = make_record()
        record = apply_average_reference(record)
        # After average reference, the mean across channels at each time point ≈ 0
        mean_across_channels = record.raw.get_data(picks="eeg").mean(axis=0)
        assert np.abs(mean_across_channels).mean() < 1e-12

    def test_no_raw_skips_gracefully(self):
        record = EEGRecord()
        record = apply_average_reference(record)
        assert record.referenced is False
        assert any("skipped" in n for n in record.processing_notes)


# ---------------------------------------------------------------------------
# Artifact detection
# ---------------------------------------------------------------------------

class TestArtifactDetection:

    def test_clean_signal_no_confounds(self):
        record = make_record()
        record.data_quality = "good"
        record = detect_artifacts(record)
        # Clean synthetic signal should have minimal confounds
        assert record.data_quality in ("good", "moderate")

    def test_muscle_artifact_detected(self):
        record = make_record()
        rng = np.random.default_rng(99)
        noise = rng.standard_normal(record.raw.n_times) * 80e-6
        for ch in ["T3", "T4"]:
            record.raw._data[record.raw.ch_names.index(ch)] += noise
        record.data_quality = "good"
        record = detect_artifacts(record)
        assert any("muscle_" in c for c in record.confounds)

    def test_drowsiness_detected(self):
        n = int(256 * 30)
        inst_freq = np.linspace(10.0, 5.5, n)
        phase = np.cumsum(2 * np.pi * inst_freq / 256)
        drifting = 30e-6 * np.sin(phase)
        record = make_record()
        for ch in ["O1", "O2"]:
            record.raw._data[record.raw.ch_names.index(ch)] = drifting
        record.data_quality = "good"
        record = detect_artifacts(record)
        assert any("drowsiness" in c for c in record.confounds)

    def test_no_raw_returns_poor(self):
        record = EEGRecord()
        record = detect_artifacts(record)
        assert record.data_quality == "poor"


# ---------------------------------------------------------------------------
# Format adapter
# ---------------------------------------------------------------------------

class TestFormatAdapter:

    def _write_fif(self):
        raw = make_raw()
        tmp = tempfile.NamedTemporaryFile(suffix="_raw.fif", delete=False)
        raw.save(tmp.name, overwrite=True, verbose=False)
        return tmp.name

    def _write_edf(self):
        raw = make_raw()
        tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
        mne.export.export_raw(tmp.name, raw, fmt="edf", overwrite=True, verbose=False)
        return tmp.name

    def _write_set(self):
        raw = make_raw()
        tmp = tempfile.NamedTemporaryFile(suffix=".set", delete=False)
        mne.export.export_raw(tmp.name, raw, fmt="eeglab", overwrite=True, verbose=False)
        return tmp.name

    def _write_csv(self):
        raw = make_raw()
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        header = ",".join(raw.ch_names)
        np.savetxt(tmp.name, raw.get_data().T * 1e6, delimiter=",",
                   header=header, comments="")
        return tmp.name, float(raw.info["sfreq"])

    def _write_npz(self):
        raw = make_raw()
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(tmp.name, data=raw.get_data(),
                 sfreq=np.array(raw.info["sfreq"]),
                 ch_names=np.array(raw.ch_names))
        return tmp.name

    def _write_mat(self):
        raw = make_raw()
        tmp = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
        scipy.io.savemat(tmp.name, {
            "data": raw.get_data() * 1e6,
            "srate": float(raw.info["sfreq"]),
            "ch_names": np.array(raw.ch_names, dtype=object),
        })
        return tmp.name

    def test_load_fif(self):
        path = self._write_fif()
        raw, meta = load_eeg(path)
        assert len(raw.ch_names) == len(EXPECTED_CHANNELS_10_20)
        assert meta["file_format"] == ".fif"

    def test_load_edf(self):
        path = self._write_edf()
        raw, meta = load_eeg(path)
        assert raw.info["sfreq"] == 256
        assert meta["file_format"] == ".edf"

    def test_load_set(self):
        path = self._write_set()
        raw, meta = load_eeg(path)
        assert len(raw.ch_names) > 0

    def test_load_csv_with_header(self):
        path, sfreq = self._write_csv()
        raw, meta = load_eeg(path, sfreq=sfreq)
        assert raw.ch_names[0] in EXPECTED_CHANNELS_10_20

    def test_load_csv_requires_sfreq(self):
        path, _ = self._write_csv()
        with pytest.raises((ValueError, Exception)):
            load_eeg(path)

    def test_load_npz(self):
        path = self._write_npz()
        raw, meta = load_eeg(path)
        assert meta["file_format"] == ".npz"
        assert len(raw.ch_names) == len(EXPECTED_CHANNELS_10_20)

    def test_load_mat(self):
        path = self._write_mat()
        raw, meta = load_eeg(path)
        assert meta["file_format"] == ".mat"
        assert raw.info["sfreq"] == 256

    def test_unsupported_format_raises(self):
        with pytest.raises(UnsupportedFormatError):
            load_eeg("recording.xyz")

    def test_legacy_csv_sfreq_kwarg(self):
        path, sfreq = self._write_csv()
        raw, _ = load_eeg(path, csv_sfreq=sfreq)
        assert raw.info["sfreq"] == sfreq

    def test_metadata_fields_present(self):
        path = self._write_fif()
        _, meta = load_eeg(path)
        for key in ("file_format", "sampling_rate", "n_channels", "channel_names"):
            assert key in meta
