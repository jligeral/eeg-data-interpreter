from preprocessor.eeg_record import EEGRecord


def apply_average_reference(record: EEGRecord) -> EEGRecord:
    """
    Apply average reference to the EEG signal.

    LEAD was trained with average-referenced data; applying this step
    aligns the recording to that training distribution.
    """
    if record.raw is None:
        record.processing_notes.append("Average reference skipped: raw signal missing.")
        return record

    try:
        record.raw.set_eeg_reference("average", projection=False, verbose=False)
        record.referenced = True
        record.processing_notes.append("Average reference applied.")
    except Exception as e:
        record.processing_notes.append(f"Average reference failed: {e}")

    return record
