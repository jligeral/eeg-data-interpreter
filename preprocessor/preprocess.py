from preprocessor.eeg_record import EEGRecord
from preprocessor.format_adapter import load_eeg
from preprocessor.channel_validation import validate_channels
from preprocessor.bandpass_filter import apply_bandpass_filter
from preprocessor.artifact_detection import detect_artifacts


def preprocess(filepath: str, csv_sfreq: float = None) -> EEGRecord:
    record = EEGRecord(filepath=filepath)

    try:
        raw, metadata = load_eeg(filepath, csv_sfreq=csv_sfreq)
        record.raw = raw
        record.file_format = metadata["file_format"]
    except Exception as e:
        record.processing_notes.append(f"Load failed: {e}")
        record.data_quality = "poor"
        return record

    record = validate_channels(record)
    record = apply_bandpass_filter(record)
    record = detect_artifacts(record)

    return record
