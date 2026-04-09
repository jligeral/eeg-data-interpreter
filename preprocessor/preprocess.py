from preprocessor.eeg_record import EEGRecord
from preprocessor.format_adapter import load_eeg
from preprocessor.channel_validation import validate_channels
from preprocessor.bandpass_filter import apply_bandpass_filter
from preprocessor.reference import apply_average_reference
from preprocessor.artifact_detection import detect_artifacts


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
    record = apply_bandpass_filter(record)
    record = apply_average_reference(record)
    record = detect_artifacts(record)

    return record
