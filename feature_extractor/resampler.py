import numpy as np
import mne
from preprocessor.eeg_record import EEGRecord

REVE_SFREQ = 200          # Hz — REVE's required sampling rate
REVE_WINDOW = 2000        # samples at 200 Hz = 10 seconds
REVE_CLIP = 15.0          # z-score clip threshold


def resample_and_window(record: EEGRecord) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Prepare EEG data for REVE:
      1. Resample to 200 Hz
      2. Slice into 2000-sample (10s) non-overlapping windows
      3. Z-score normalize each window, clip to ±15

    Returns:
        windows       : np.ndarray, shape (n_windows, n_channels, 2000), float32
        channel_names : list of channel name strings (in order)
        notes         : processing notes
    """
    notes = []

    if record.raw is None:
        notes.append("Resampler skipped: raw signal missing.")
        return np.empty((0,), dtype=np.float32), [], notes

    raw = record.raw.copy()

    # Resample if needed
    current_sfreq = float(raw.info["sfreq"])
    if current_sfreq != REVE_SFREQ:
        raw.resample(REVE_SFREQ)
        notes.append(f"Resampled from {current_sfreq:.1f} Hz to {REVE_SFREQ} Hz for REVE.")
    else:
        notes.append(f"Sampling rate already {REVE_SFREQ} Hz — no resampling needed.")

    data = raw.get_data(picks="eeg")  # (n_channels, n_samples)
    channel_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    n_channels, n_samples = data.shape

    if n_samples < REVE_WINDOW:
        notes.append(
            f"Recording too short for REVE ({n_samples} samples < {REVE_WINDOW} required). "
            f"Need at least {REVE_WINDOW / REVE_SFREQ:.0f}s of data."
        )
        return np.empty((0,), dtype=np.float32), channel_names, notes

    # Slice into non-overlapping windows
    n_windows = n_samples // REVE_WINDOW
    trimmed = data[:, : n_windows * REVE_WINDOW]
    windows = trimmed.reshape(n_channels, n_windows, REVE_WINDOW)
    windows = windows.transpose(1, 0, 2)  # (n_windows, n_channels, 2000)

    # Z-score normalize per window, clip to ±15
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)  # avoid divide-by-zero on flat channels
    windows = (windows - mean) / std
    windows = np.clip(windows, -REVE_CLIP, REVE_CLIP)

    notes.append(
        f"Windowed into {n_windows} × {REVE_WINDOW}-sample windows "
        f"({REVE_WINDOW / REVE_SFREQ:.0f}s each). Z-scored and clipped to ±{REVE_CLIP}."
    )

    return windows.astype(np.float32), channel_names, notes
