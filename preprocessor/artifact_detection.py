import mne
import numpy as np
from mne.preprocessing import ICA
from preprocessor.eeg_record import EEGRecord

MUSCLE_FREQ_BAND = (20, 40)          # Hz
MUSCLE_POWER_THRESHOLD = 2.5         # z-score above channel mean
DROWSINESS_ALPHA_THRESHOLD = 7.5     # Hz
MIN_ICA_COMPONENTS = 2
MAX_ICA_COMPONENTS = 15


def detect_artifacts(record: EEGRecord) -> EEGRecord:
    if record.raw is None:
        record.processing_notes.append(
            "Artifact detection skipped: raw signal missing."
        )
        record.data_quality = "poor"
        return record

    raw_working = record.raw.copy()
    confounds = []

    ocular_severity, raw_after_ocular = detect_ocular(raw_working, record)
    if ocular_severity is not None:
        confounds.append(f"ocular_{ocular_severity}")
    record.raw = raw_after_ocular

    muscle_severity, muscle_regions = detect_muscle(record.raw)
    if muscle_severity is not None:
        flag = f"muscle_{muscle_severity}"
        if muscle_regions:
            flag += f":{','.join(muscle_regions)}"
        confounds.append(flag)

    bad_channels, elec_severity = detect_electrode_artifacts(record.raw)
    if elec_severity is not None:
        flag = f"electrode_{elec_severity}"
        if bad_channels:
            flag += f":{','.join(bad_channels)}"
        confounds.append(flag)

        existing_bads = list(record.raw.info.get("bads", []))
        merged = []
        seen = set()
        for ch in existing_bads + bad_channels:
            if ch not in seen:
                seen.add(ch)
                merged.append(ch)
        record.raw.info["bads"] = merged

    drowsy_severity, drowsy_detail = detect_drowsiness(record.raw, record)
    if drowsy_severity is not None:
        flag = f"drowsiness_{drowsy_severity}"
        if drowsy_detail:
            flag += f":{drowsy_detail}"
        confounds.append(flag)

    existing_confounds = list(record.confounds)
    for c in confounds:
        if c not in existing_confounds:
            existing_confounds.append(c)
    record.confounds = existing_confounds

    record.data_quality = _derive_quality(record.data_quality, confounds)

    if confounds:
        record.processing_notes.append(
            f"Artifact detection added confounds: {confounds}"
        )
    else:
        record.processing_notes.append(
            "Artifact detection found no major confounds."
        )

    return record


def detect_ocular(raw, record: EEGRecord) -> tuple[str | None, mne.io.BaseRaw]:
    """ICA-based ocular artifact detection and removal when feasible."""
    try:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        n_eeg = len(eeg_picks)

        if n_eeg < MIN_ICA_COMPONENTS:
            record.processing_notes.append(
                "Ocular artifact check skipped: not enough EEG channels for ICA."
            )
            return None, raw

        n_components = min(MAX_ICA_COMPONENTS, max(MIN_ICA_COMPONENTS, n_eeg - 1))
        ica = ICA(n_components=n_components, random_state=42, max_iter=800)
        ica.fit(raw, picks="eeg")

        eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]

        frontal_fallbacks = [ch for ch in raw.ch_names if ch.upper() in {"FP1", "FP2", "F7", "F8"}]

        if eog_channels:
            eog_idx, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels[0])
            ref_name = eog_channels[0]
        elif frontal_fallbacks:
            eog_idx, eog_scores = ica.find_bads_eog(raw, ch_name=frontal_fallbacks[0])
            ref_name = frontal_fallbacks[0]
            record.processing_notes.append(
                f"Ocular detection used frontal fallback channel: {ref_name}"
            )
        else:
            record.processing_notes.append(
                "Ocular artifact check skipped: no EOG or frontal proxy channel available."
            )
            return None, raw

        if not eog_idx:
            return None, raw

        max_score = float(np.max(np.abs(np.asarray(eog_scores)[eog_idx])))

        if max_score > 0.9 or len(eog_idx) > 2:
            severity = "severe"
        elif max_score > 0.6:
            severity = "moderate"
        else:
            severity = "mild"

        ica.exclude = list(eog_idx)
        raw_clean = raw.copy()
        ica.apply(raw_clean)

        record.processing_notes.append(
            f"ICA removed ocular-related components {list(eog_idx)} using reference {ref_name}."
        )
        return severity, raw_clean

    except Exception as e:
        record.processing_notes.append(
            f"ICA failed; ocular artifact could not be assessed/removed: {e}"
        )
        return None, raw


def detect_muscle(raw) -> tuple[str | None, list[str]]:
    """Detect possible muscle artifact from elevated 20–40 Hz power."""
    from scipy.signal import welch

    data = raw.get_data(picks="eeg")
    sfreq = float(raw.info["sfreq"])

    if data.size == 0:
        return None, []

    nperseg = max(8, min(data.shape[1], int(sfreq * 2)))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)

    muscle_idx = np.where(
        (freqs >= MUSCLE_FREQ_BAND[0]) & (freqs <= MUSCLE_FREQ_BAND[1])
    )[0]

    if len(muscle_idx) == 0:
        return None, []

    muscle_power = psd[:, muscle_idx].mean(axis=1)
    std = float(muscle_power.std())
    if std == 0:
        return None, []

    z_scores = (muscle_power - muscle_power.mean()) / std
    affected_idx = np.where(z_scores > MUSCLE_POWER_THRESHOLD)[0]

    if len(affected_idx) == 0:
        return None, []

    affected_names = [raw.ch_names[i] for i in affected_idx]
    affected_upper = [ch.upper() for ch in affected_names]

    temporal_markers = {"T3", "T4", "T5", "T6", "T7", "T8", "P7", "P8"}
    frontal_markers = {"F7", "F8", "FP1", "FP2"}

    regions = []
    if any(ch in temporal_markers for ch in affected_upper):
        regions.append("temporal")
    if any(ch in frontal_markers for ch in affected_upper):
        regions.append("frontal")

    severity = (
        "severe" if len(affected_idx) > 4
        else "moderate" if len(affected_idx) > 1
        else "mild"
    )
    return severity, regions


def detect_electrode_artifacts(raw) -> tuple[list[str], str | None]:
    """Detect channels with unusually low/high variability."""
    data = raw.get_data(picks="eeg")

    if data.size == 0:
        return [], None

    channel_std = np.std(data, axis=1)
    std_of_std = float(channel_std.std())
    if std_of_std == 0:
        return [], None

    z_scores = (channel_std - channel_std.mean()) / std_of_std
    bad_idx = np.where(np.abs(z_scores) > 3.0)[0]
    bad_channels = [raw.ch_names[i] for i in bad_idx]

    if not bad_channels:
        return [], None

    severity = (
        "severe" if len(bad_channels) > 3
        else "moderate" if len(bad_channels) > 1
        else "mild"
    )
    return bad_channels, severity


def detect_drowsiness(raw, record: EEGRecord) -> tuple[str | None, str | None]:
    """Heuristic drowsiness detection from posterior alpha slowing."""
    from scipy.signal import welch

    present = {ch.upper(): ch for ch in raw.ch_names}
    needed = [ch for ch in ["O1", "O2"] if ch in present]

    if len(needed) < 2:
        record.processing_notes.append(
            "Drowsiness check skipped: O1/O2 not both available."
        )
        return None, None

    try:
        actual_names = [present["O1"], present["O2"]]
        data = raw.get_data(picks=actual_names)
        sfreq = float(raw.info["sfreq"])
        n_samples = data.shape[1]

        if n_samples < int(sfreq * 8):
            record.processing_notes.append(
                "Drowsiness check skipped: recording too short for stable posterior alpha estimate."
            )
            return None, None

        third = n_samples // 3
        early = data[:, :third]
        late = data[:, -third:]

        def peak_freq(segment):
            nperseg = max(8, min(segment.shape[1], int(sfreq * 4)))
            freqs, psd = welch(segment, fs=sfreq, nperseg=nperseg)
            alpha_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
            if len(alpha_idx) == 0:
                return None
            avg_psd = psd[:, alpha_idx].mean(axis=0)
            return float(freqs[alpha_idx[np.argmax(avg_psd)]])

        early_pdr = peak_freq(early)
        late_pdr = peak_freq(late)

        if early_pdr is None or late_pdr is None:
            return None, None

        drift = early_pdr - late_pdr

        if late_pdr < DROWSINESS_ALPHA_THRESHOLD:
            severity = "severe" if late_pdr < 6.5 else "moderate"
            detail = "late_recording" if drift > 0.5 else "throughout"
            return severity, detail

        if drift > 1.0:
            return "mild", "late_recording"

        return None, None

    except Exception as e:
        record.processing_notes.append(
            f"Drowsiness check failed: {e}"
        )
        return None, None


def _derive_quality(current_quality: str, confounds: list[str]) -> str:
    """Escalate data quality rating based on confound severities."""
    if any("severe" in c for c in confounds):
        return "poor"
    if any("moderate" in c for c in confounds):
        return "poor" if current_quality == "poor" else "moderate"
    if any("mild" in c for c in confounds):
        if current_quality in {"poor", "moderate"}:
            return current_quality
        return "moderate"
    return current_quality if current_quality != "unknown" else "good"
