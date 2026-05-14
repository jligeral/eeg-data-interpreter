import numpy as np

from preprocessor.eeg_record import EEGRecord
from preprocessor.format_adapter import load_eeg
from preprocessor.channel_validation import validate_channels
from preprocessor.bandpass_filter import apply_bandpass_filter
from preprocessor.reference import apply_average_reference
from preprocessor.artifact_detection import detect_artifacts

_SNAPSHOT_DURATION = 15.0   # seconds
_SNAPSHOT_TARGET_HZ = 250.0 # max sampling rate stored


def _make_snapshot(raw) -> dict:
    """Return a lightweight dict snapshot of the first _SNAPSHOT_DURATION seconds."""
    try:
        import mne
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if not picks.size:
            picks = list(range(len(raw.ch_names)))
        ch_names = [raw.ch_names[i] for i in picks]
        sfreq = float(raw.info["sfreq"])
        n_samples = min(int(_SNAPSHOT_DURATION * sfreq), raw.n_times)
        data_uv = raw.get_data(picks=picks, start=0, stop=n_samples) * 1e6
        times = raw.times[:n_samples]
        # Decimate if sampling rate is higher than target
        if sfreq > _SNAPSHOT_TARGET_HZ:
            step = int(sfreq / _SNAPSHOT_TARGET_HZ)
            data_uv = data_uv[:, ::step]
            times = times[::step]
        return {
            "channels": ch_names,
            "sfreq": min(sfreq, _SNAPSHOT_TARGET_HZ),
            "times": times.tolist(),
            "data": data_uv.tolist(),
        }
    except Exception:
        return {}


def preprocess(
    filepath: str,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    # Legacy alias
    csv_sfreq: float | None = None,
) -> EEGRecord:
    """
    Full preprocessing pipeline.

      1. Load file (all common EEG formats supported)
      2. Validate and canonicalize channel names
      3. Bandpass filter 0.5–45 Hz
      4. Average reference
      5. Artifact detection

    Parameters
    ----------
    filepath  : path to EEG file
    sfreq     : sampling rate (required for .csv / .tsv / .txt / .npy)
    ch_names  : channel names (optional, used for .npy)
    csv_sfreq : legacy alias for sfreq
    """
    if csv_sfreq is not None and sfreq is None:
        sfreq = csv_sfreq

    record = EEGRecord(filepath=filepath)

    try:
        raw, metadata = load_eeg(filepath, sfreq=sfreq, ch_names=ch_names)
        record.raw = raw
        record.file_format = metadata["file_format"]
    except Exception as e:
        record.processing_notes.append(f"Load failed: {e}")
        record.data_quality = "poor"
        return record

    record = validate_channels(record)
    if record.raw is not None:
        record.snapshot_pre = _make_snapshot(record.raw)

    record = apply_bandpass_filter(record)
    record = apply_average_reference(record)
    record = detect_artifacts(record)

    if record.raw is not None:
        record.snapshot_post = _make_snapshot(record.raw)

    return record
