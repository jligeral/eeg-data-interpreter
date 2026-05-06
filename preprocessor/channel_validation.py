import mne
import numpy as np
from preprocessor.eeg_record import EEGRecord

# Standard 19-channel 10-20 montage — matches LEAD's expected input
EXPECTED_CHANNELS_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]

POSTERIOR_CHANNELS = ["O1", "O2", "P3", "P4", "T5", "T6"]

# Common alternative names → canonical 10-20 names
CHANNEL_ALIASES = {
    "T7": "T3", "T8": "T4",
    "P7": "T5", "P8": "T6",
}

# Characters that manufacturers append but carry no semantic meaning
_STRIP_PATTERNS = ["-REF", "-LE", "-AVG", " REF", "_REF", "EEG ", " "]

MIN_SAMPLING_RATE_HZ = 80
FLAT_VARIANCE_THRESHOLD_UV2 = 1.0
HIGH_AMPLITUDE_THRESHOLD_UV = 500.0


def canonicalize_channel(name: str) -> str:
    clean = name.strip()
    for pat in _STRIP_PATTERNS:
        clean = clean.replace(pat, "")
    clean = clean.replace("_", "")
    return CHANNEL_ALIASES.get(clean, clean)


def validate_channels(record: EEGRecord) -> EEGRecord:
    raw = record.raw
    if raw is None:
        record.processing_notes.append("Validation failed: raw signal is missing.")
        record.data_quality = "poor"
        return record

    # Build canonical name map and rename in-place
    rename_map = {}
    for ch in raw.ch_names:
        canonical = canonicalize_channel(ch)
        if canonical != ch:
            rename_map[ch] = canonical
    if rename_map:
        raw.rename_channels(rename_map)
        record.processing_notes.append(
            f"Renamed channels: {rename_map}"
        )

    canonical_set = {ch.upper() for ch in raw.ch_names}
    expected_upper = {ch.upper(): ch for ch in EXPECTED_CHANNELS_10_20}

    missing = [exp for up, exp in expected_upper.items() if up not in canonical_set]
    unknown = [ch for ch in raw.ch_names if ch.upper() not in expected_upper]
    missing_posterior = [ch for ch in POSTERIOR_CHANNELS if ch.upper() not in canonical_set]

    sfreq = float(raw.info["sfreq"])
    rate_adequate = sfreq >= MIN_SAMPLING_RATE_HZ

    data_uv = raw.get_data(picks="eeg") * 1e6
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
    record.unknown_channels = unknown
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
        confounds.append(f"sampling_rate_low:{sfreq:.0f}Hz")

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
