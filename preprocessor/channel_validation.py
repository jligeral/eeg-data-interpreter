import re

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

# EGI HydroCel Geodesic Sensor Net montage names by max electrode count
_EGI_MONTAGES = {
    64:  "GSN-HydroCel-64_1.0",
    128: "GSN-HydroCel-128",
    256: "GSN-HydroCel-256",
}


def canonicalize_channel(name: str) -> str:
    clean = name.strip()
    for pat in _STRIP_PATTERNS:
        clean = clean.replace(pat, "")
    clean = clean.replace("_", "")
    return CHANNEL_ALIASES.get(clean, clean)


def _detect_egi(ch_names: list[str]) -> int | None:
    """
    Return the max electrode number if ≥80% of channels look like EGI E-notation
    (E1, E2, …), otherwise None.
    """
    nums = []
    for ch in ch_names:
        m = re.match(r"^E(\d+)$", ch, re.IGNORECASE)
        if m:
            nums.append(int(m.group(1)))
    if len(nums) < len(ch_names) * 0.8:
        return None
    return max(nums)


def _build_egi_rename_map(ch_names: list[str], max_num: int) -> dict[str, str]:
    """
    Nearest-neighbour map from EGI electrode names to standard 10-20 names,
    using MNE's built-in EGI montage positions as the reference geometry.

    Returns a dict {egi_name: standard_10_20_name} covering as many of the
    19 expected channels as the EGI layout can support.
    """
    # Pick the smallest EGI montage that covers the electrode count
    montage_name = None
    for size in sorted(_EGI_MONTAGES):
        if max_num <= size:
            montage_name = _EGI_MONTAGES[size]
            break
    if montage_name is None:
        return {}

    try:
        egi_montage = mne.channels.make_standard_montage(montage_name)
        std_montage = mne.channels.make_standard_montage("standard_1020")
    except Exception:
        return {}

    egi_pos_dict = egi_montage.get_positions()["ch_pos"]
    std_pos_dict = std_montage.get_positions()["ch_pos"]

    # Only work with EGI channels that are actually present in this file
    present = [ch for ch in ch_names if ch in egi_pos_dict]
    if not present:
        return {}

    targets = [ch for ch in EXPECTED_CHANNELS_10_20 if ch in std_pos_dict]
    if not targets:
        return {}

    present_pos = np.array([egi_pos_dict[ch] for ch in present])   # (n_egi, 3)
    target_pos  = np.array([std_pos_dict[ch]  for ch in targets])   # (n_std, 3)

    # For each target 10-20 channel find the closest EGI electrode
    dists = np.linalg.norm(
        target_pos[:, None, :] - present_pos[None, :, :], axis=-1
    )  # (n_std, n_egi)

    rename_map: dict[str, str] = {}
    used: set[str] = set()
    # Assign greedily in order of increasing distance to keep best matches first
    order = np.argsort(dists.min(axis=1))  # std channels sorted by their best match distance
    for std_idx in order:
        egi_idx = int(np.argmin(dists[std_idx]))
        egi_name = present[egi_idx]
        if egi_name not in used:
            rename_map[egi_name] = targets[std_idx]
            used.add(egi_name)

    return rename_map


def validate_channels(record: EEGRecord) -> EEGRecord:
    raw = record.raw
    if raw is None:
        record.processing_notes.append("Validation failed: raw signal is missing.")
        record.data_quality = "poor"
        return record

    # EGI E-notation → 10-20 mapping (must run before generic canonicalization)
    max_egi = _detect_egi(raw.ch_names)
    if max_egi is not None:
        egi_map = _build_egi_rename_map(raw.ch_names, max_egi)
        if egi_map:
            raw.rename_channels(egi_map)
            record.processing_notes.append(
                f"EGI montage detected (max E{max_egi}): mapped "
                f"{len(egi_map)} electrodes to 10-20 names."
            )
        else:
            record.processing_notes.append(
                f"EGI montage detected (max E{max_egi}) but position mapping failed."
            )

    # Generic canonicalization (strip suffixes, apply aliases)
    rename_map = {}
    for ch in raw.ch_names:
        canonical = canonicalize_channel(ch)
        if canonical != ch:
            rename_map[ch] = canonical
    if rename_map:
        raw.rename_channels(rename_map)
        record.processing_notes.append(f"Renamed channels: {rename_map}")

    # Case-correct any channel whose uppercase matches a 10-20 name
    # (e.g. FZ → Fz, CZ → Cz, FP1 → Fp1)
    _upper_to_10_20 = {ch.upper(): ch for ch in EXPECTED_CHANNELS_10_20}
    case_map = {
        ch: _upper_to_10_20[ch.upper()]
        for ch in raw.ch_names
        if ch.upper() in _upper_to_10_20 and ch != _upper_to_10_20[ch.upper()]
    }
    if case_map:
        raw.rename_channels(case_map)
        record.processing_notes.append(f"Corrected channel casing: {case_map}")

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
