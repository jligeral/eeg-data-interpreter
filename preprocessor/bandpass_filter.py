from preprocessor.eeg_record import EEGRecord

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 45.0  # LEAD trained on 0.5–45 Hz


def apply_bandpass_filter(record: EEGRecord) -> EEGRecord:
    if record.raw is None:
        record.processing_notes.append("Bandpass filter skipped: raw signal missing.")
        record.data_quality = "poor"
        return record

    try:
        record.raw.load_data(verbose=False)
        sfreq = float(record.raw.info["sfreq"])
        record.sampling_rate = sfreq

        nyquist = sfreq / 2
        high_cut = min(HIGH_CUTOFF_HZ, nyquist * 0.9)
        if high_cut < HIGH_CUTOFF_HZ:
            record.processing_notes.append(
                f"High cutoff adjusted {HIGH_CUTOFF_HZ} → {high_cut:.1f} Hz "
                f"(sampling rate {sfreq:.0f} Hz)."
            )

        record.raw.filter(
            l_freq=LOW_CUTOFF_HZ,
            h_freq=high_cut,
            method="fir",
            fir_window="hamming",
            phase="zero",
            picks="eeg",
            verbose=False,
        )
        record.processing_notes.append(
            f"Bandpass filter applied: {LOW_CUTOFF_HZ}–{high_cut:.1f} Hz, "
            f"FIR zero-phase Hamming window."
        )

    except Exception as e:
        record.processing_notes.append(f"Bandpass filter failed: {e}")
        record.data_quality = "poor"

    return record
