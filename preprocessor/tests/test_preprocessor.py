import numpy as np
import pytest
import mne

from preprocessor.eeg_record import EEGRecord
from preprocessor.channel_validation import validate_channels, EXPECTED_CHANNELS_10_20, POSTERIOR_CHANNELS
from preprocessor.bandpass_filter import apply_bandpass_filter
from preprocessor.artifact_detection import detect_artifacts, detect_muscle, detect_electrode_artifacts, detect_drowsiness, _derive_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw(ch_names, sfreq=256, duration=10, amplitude_uv=10.0):
    """Build a synthetic MNE RawArray with white-noise EEG data."""
    n_samples = int(sfreq * duration)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(ch_names), n_samples)) * amplitude_uv * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def make_record(ch_names=None, sfreq=256, duration=10, amplitude_uv=10.0):
    if ch_names is None:
        ch_names = list(EXPECTED_CHANNELS_10_20)
    raw = make_raw(ch_names, sfreq=sfreq, duration=duration, amplitude_uv=amplitude_uv)
    record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
    return record


# ---------------------------------------------------------------------------
# channel_validation tests
# ---------------------------------------------------------------------------

class TestValidateChannels:

    def test_full_montage_is_good(self):
        record = make_record(ch_names=list(EXPECTED_CHANNELS_10_20))
        result = validate_channels(record)
        assert result.data_quality == "good"
        assert result.missing_channels == []
        assert result.confounds == []

    def test_missing_posterior_channels_is_poor(self):
        ch_names = [ch for ch in EXPECTED_CHANNELS_10_20 if ch not in POSTERIOR_CHANNELS]
        record = make_record(ch_names=ch_names)
        result = validate_channels(record)
        assert result.data_quality == "poor"
        assert any("missing_posterior_channels" in c for c in result.confounds)

    def test_missing_non_posterior_channel_is_moderate(self):
        # Remove Fz (frontal, not posterior)
        ch_names = [ch for ch in EXPECTED_CHANNELS_10_20 if ch != "Fz"]
        record = make_record(ch_names=ch_names)
        result = validate_channels(record)
        assert result.data_quality in ("moderate", "good")
        assert "Fz" in result.missing_channels

    def test_flat_channel_detected(self):
        ch_names = list(EXPECTED_CHANNELS_10_20)
        raw = make_raw(ch_names)
        # Zero out one channel to make it flat
        raw._data[0, :] = 0.0
        record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
        result = validate_channels(record)
        assert len(result.flat_channels) >= 1
        assert any("flat_channels" in c for c in result.confounds)
        assert result.data_quality in ("moderate", "poor")

    def test_high_amplitude_channel_detected(self):
        ch_names = list(EXPECTED_CHANNELS_10_20)
        raw = make_raw(ch_names)
        # Spike one channel way above 500 µV threshold
        raw._data[0, :] = 600e-6
        record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
        result = validate_channels(record)
        assert len(result.high_amplitude_channels) >= 1
        assert any("high_amplitude_channels" in c for c in result.confounds)

    def test_low_sampling_rate_is_poor(self):
        record = make_record(sfreq=64)  # below 80 Hz threshold
        result = validate_channels(record)
        assert result.data_quality == "poor"
        assert any("sampling_rate_low" in c for c in result.confounds)

    def test_channel_alias_resolved(self):
        # T7/T8/P7/P8 are aliases for T3/T4/T5/T6 — should not appear as missing
        ch_names = [ch for ch in EXPECTED_CHANNELS_10_20 if ch not in ("T3", "T4", "T5", "T6")]
        ch_names += ["T7", "T8", "P7", "P8"]
        record = make_record(ch_names=ch_names)
        result = validate_channels(record)
        for alias_target in ("T3", "T4", "T5", "T6"):
            assert alias_target not in result.missing_channels

    def test_no_raw_is_poor(self):
        record = EEGRecord()
        result = validate_channels(record)
        assert result.data_quality == "poor"
        assert any("missing" in note.lower() for note in result.processing_notes)

    def test_structured_fields_populated(self):
        record = make_record()
        result = validate_channels(record)
        assert isinstance(result.flat_channels, list)
        assert isinstance(result.high_amplitude_channels, list)
        assert isinstance(result.montage_ok, bool)


# ---------------------------------------------------------------------------
# bandpass_filter tests
# ---------------------------------------------------------------------------

class TestApplyBandpassFilter:

    def test_filter_runs_on_valid_record(self):
        record = make_record(sfreq=256)
        result = apply_bandpass_filter(record)
        assert any("Bandpass filter applied" in note for note in result.processing_notes)

    def test_filter_skipped_when_no_raw(self):
        record = EEGRecord()
        result = apply_bandpass_filter(record)
        assert result.data_quality == "poor"
        assert any("skipped" in note for note in result.processing_notes)

    def test_high_cutoff_adjusted_for_low_sfreq(self):
        # sfreq=64 → nyquist=32, which is below 40 Hz high cutoff
        record = make_record(sfreq=64)
        result = apply_bandpass_filter(record)
        assert any("adjusted" in note for note in result.processing_notes)

    def test_sampling_rate_recorded(self):
        record = make_record(sfreq=256)
        result = apply_bandpass_filter(record)
        assert result.sampling_rate == 256.0


# ---------------------------------------------------------------------------
# EEGRecord tests
# ---------------------------------------------------------------------------

class TestEEGRecord:

    def test_default_fields(self):
        record = EEGRecord()
        assert record.data_quality == "unknown"
        assert record.confounds == []
        assert record.processing_notes == []
        assert record.flat_channels == []
        assert record.high_amplitude_channels == []
        assert record.montage_ok is True

    def test_processing_notes_accumulate(self):
        record = EEGRecord()
        record = validate_channels(record)
        record = apply_bandpass_filter(record)
        assert len(record.processing_notes) >= 2


# ---------------------------------------------------------------------------
# artifact_detection tests
# ---------------------------------------------------------------------------

class TestDetectArtifacts:

    def test_no_raw_is_poor(self):
        record = EEGRecord()
        result = detect_artifacts(record)
        assert result.data_quality == "poor"
        assert any("skipped" in note for note in result.processing_notes)

    def test_clean_signal_adds_no_confounds(self):
        # Build a signal with a stable 10 Hz alpha oscillation so drowsiness
        # detector sees a consistent peak across early and late recording thirds.
        ch_names = list(EXPECTED_CHANNELS_10_20)
        sfreq = 256
        duration = 30  # needs to be long enough for stable Welch estimates
        n_samples = int(sfreq * duration)
        t = np.linspace(0, duration, n_samples)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((len(ch_names), n_samples)) * 5e-6
        # Add stable 10 Hz alpha to O1 and O2
        for ch in ["O1", "O2"]:
            idx = ch_names.index(ch)
            data[idx] += 20e-6 * np.sin(2 * np.pi * 10 * t)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
        result = detect_artifacts(record)
        artifact_confounds = [c for c in result.confounds
                              if c.startswith(("ocular_", "muscle_", "drowsiness_"))]
        assert artifact_confounds == []
        assert any("no major confounds" in note for note in result.processing_notes)

    def test_muscle_artifact_detected(self):
        ch_names = list(EXPECTED_CHANNELS_10_20)
        raw = make_raw(ch_names, sfreq=256)
        # Inject high-frequency noise into temporal channels to simulate muscle artifact
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10, 256 * 10)
        muscle_signal = rng.standard_normal(len(t)) * 50e-6  # large amplitude
        for ch in ["T3", "T4"]:
            idx = raw.ch_names.index(ch)
            raw._data[idx] += muscle_signal
        record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
        result = detect_artifacts(record)
        assert any("muscle_" in c for c in result.confounds)

    def test_electrode_artifact_detected(self):
        ch_names = list(EXPECTED_CHANNELS_10_20)
        raw = make_raw(ch_names, sfreq=256)
        # Make one channel have massive std — far above z-score threshold of 3
        raw._data[0] *= 1000
        record = EEGRecord(raw=raw, filepath="synthetic", file_format=".fif")
        result = detect_artifacts(record)
        assert any("electrode_" in c for c in result.confounds)

    def test_confounds_not_duplicated(self):
        record = make_record(sfreq=256)
        record.confounds = ["muscle_mild:temporal"]
        result = detect_artifacts(record)
        assert result.confounds.count("muscle_mild:temporal") == 1

    def test_raw_replaced_after_ocular_ica(self):
        # After detect_artifacts, record.raw should still be a valid Raw object
        record = make_record(sfreq=256)
        original_id = id(record.raw)
        result = detect_artifacts(record)
        assert result.raw is not None


class TestDeriveQuality:

    def test_severe_confound_always_poor(self):
        assert _derive_quality("good", ["ocular_severe"]) == "poor"
        assert _derive_quality("moderate", ["ocular_severe"]) == "poor"

    def test_moderate_confound_escalates_good_to_moderate(self):
        assert _derive_quality("good", ["muscle_moderate"]) == "moderate"

    def test_moderate_confound_keeps_poor_as_poor(self):
        assert _derive_quality("poor", ["muscle_moderate"]) == "poor"

    def test_mild_confound_escalates_good_to_moderate(self):
        assert _derive_quality("good", ["electrode_mild"]) == "moderate"

    def test_mild_confound_preserves_poor(self):
        assert _derive_quality("poor", ["electrode_mild"]) == "poor"

    def test_no_confounds_preserves_quality(self):
        assert _derive_quality("good", []) == "good"
        assert _derive_quality("poor", []) == "poor"
