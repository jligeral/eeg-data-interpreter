from preprocessor.eeg_record import EEGRecord

def apply_bandpass_filter(record: EEGRecord) -> EEGRecord:
    LOW_CUTOFF_HZ = 0.5
    HIGH_CUTOFF_HZ = 40.0

    if record.raw is None:
        record.processing_notes.append("Bandpass filter skipped: raw signal missing.")
        record.data_quality = "poor"
        return record

    try:
        record.raw.load_data()
        sfreq = float(record.raw.info["sfreq"])
        record.sampling_rate = sfreq

        nyquist = sfreq / 2
        if HIGH_CUTOFF_HZ >= nyquist:
            adjusted_high = nyquist * 0.9
            record.processing_notes.append(
                f"High cutoff adjusted from {HIGH_CUTOFF_HZ} to "
                f"{adjusted_high:.1f} Hz — sampling rate only "
                f"{sfreq} Hz"
            )
            high_cut = adjusted_high
        else:
            high_cut = HIGH_CUTOFF_HZ

        record.raw.filter(
            l_freq=LOW_CUTOFF_HZ,
            h_freq=high_cut,
            method="fir",
            fir_window="hamming",
            phase="zero",
            picks="eeg",
        )

        record.processing_notes.append(
            f"Bandpass filter applied: "
            f"{LOW_CUTOFF_HZ}–{high_cut} Hz, "
            f"FIR zero-phase Hamming window"
        )

    except Exception as e:
        record.processing_notes.append(f"Bandpass filter failed: {e}")
        record.data_quality = "poor"

    return record
