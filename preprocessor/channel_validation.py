import mne
import numpy as np
from preprocessor.eeg_record import EEGRecord

EXPECTED_CHANNELS_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]

POSTERIOR_CHANNELS = ["O1", "O2", "P3", "P4", "T5", "T6"]

CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

MIN_SAMPLING_RATE_HZ = 80
FLAT_VARIANCE_THRESHOLD_UV2 = 1.0
HIGH_AMPLITUDE_THRESHOLD_UV = 500.0


def canonicalize_channel(name: str) -> str:
    clean = name.strip().replace("EEG ", "").replace("-REF", "").replace("_", "")
    clean = clean.replace(" ", "")
    return CHANNEL_ALIASES.get(clean, clean)


def validate_channels(record: EEGRecord) -> EEGRecord:
    raw = record.raw
    if raw is None:
        record.processing_notes.append("Validation failed: raw signal is missing.")
        record.data_quality = "poor"
        return record

    canonical_labels = [canonicalize_channel(ch) for ch in raw.ch_names]
    file_channels_upper = {ch.upper() for ch in canonical_labels}

    missing = [
        ch for ch in EXPECTED_CHANNELS_10_20
        if ch.upper() not in file_channels_upper
    ]
    unexpected = [
        ch for ch in canonical_labels
        if ch.upper() not in {exp.upper() for exp in EXPECTED_CHANNELS_10_20}
    ]
    missing_posterior = [
        ch for ch in POSTERIOR_CHANNELS
        if ch.upper() not in file_channels_upper
    ]

    sfreq = float(raw.info["sfreq"])
    rate_adequate = sfreq >= MIN_SAMPLING_RATE_HZ

    # Convert to microvolts for threshold checks
    data_uv = raw.get_data() * 1e6
    channel_variance_uv2 = np.var(data_uv, axis=1)

    flat_channels = [
        raw.ch_names[i]
        for i, v in enumerate(channel_variance_uv2)
        if v < FLAT_VARIANCE_THRESHOLD_UV2
    ]

    high_amp_channels = [
        raw.ch_names[i]
        for i, v in enumerate(np.max(np.abs(data_uv), axis=1))
        if v > HIGH_AMPLITUDE_THRESHOLD_UV
    ]

    montage_ok = True
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.copy().set_montage(montage, match_case=False, on_missing="warn")
    except Exception as e:
        montage_ok = False
        record.processing_notes.append(f"Montage compatibility check failed: {e}")

    record.sampling_rate = sfreq
    record.n_channels = len(raw.ch_names)
    record.channel_labels = list(raw.ch_names)
    record.missing_channels = missing
    record.unknown_channels = unexpected

    record.flat_channels = flat_channels
    record.high_amplitude_channels = high_amp_channels
    record.montage_ok = montage_ok

    confounds = []

    if missing_posterior:
        confounds.append(f"missing_posterior_channels:{','.join(missing_posterior)}")

    if flat_channels:
        confounds.append(f"flat_channels:{','.join(flat_channels)}")

    if high_amp_channels:
        confounds.append(f"high_amplitude_channels:{','.join(high_amp_channels)}")

    if not rate_adequate:
        confounds.append(f"sampling_rate_low:{sfreq}Hz")

    record.confounds.extend(confounds)

    if missing and not missing_posterior:
        record.processing_notes.append(f"Non-critical channels absent: {missing}")

    if missing_posterior or not rate_adequate:
        record.data_quality = "poor"
    elif missing or flat_channels or high_amp_channels or not montage_ok:
        record.data_quality = "moderate"
    else:
        record.data_quality = "good"

    return record
